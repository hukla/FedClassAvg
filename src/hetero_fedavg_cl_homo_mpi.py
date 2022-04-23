from collections import OrderedDict
import copy
import os
import numpy as np
import string
import time
import random
import math

import torch
import torch.nn as nn 
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import wandb

from argparse import ArgumentParser

from mpi4py import MPI
from PIL import Image
from datetime import datetime
import utils
from utils import partition_data, EarlyStopping
import supcon_utils
from losses import SupConLoss


### Args
parser = ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--target_class_label_idx', type=int, default=0, help='for navon')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--output_path', type=str, default=None, help="path to write output files")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--runfile', type=str, default=None)

### FL-related paprams ###
parser.add_argument('--c', type=float, default=.1,
                    help='Fraction of participating clients at each communication round (default: 0.1)')
parser.add_argument('--max_rounds', type=int, default=1000)
parser.add_argument('--local_epochs', type=int, default=1)
parser.add_argument('--virtual_per_node', type=int, default=10,
                    help='how many virtual nodes to iterate in each mpi process (default: 1)')
parser.add_argument('--data_partition', type=str, default='noniid-labeldir', help='iid|noniid-labelskew|noniid-labeldir')
parser.add_argument('--beta', type=float, default=0.5, help='The concentration parameter of the Dirichlet distribution for heterogeneous partition.')
parser.add_argument('--models', type=str, default='resnet')

# parser.add_argument('--print-weight', type=bool, default=False,
                    # help='Whether to print weight and gradients (for debug, default: False)')
parser.add_argument('--epochs', type=int, default=1,
                    help='Client update epochs at every communication round (default: 1)')
parser.add_argument('--mu', type=float, default=0.1,
                    help='Hyperparameter for Proximal term. FedAvg if 0. (default: 0)')
parser.add_argument('--is_experiment', type=bool, default=False,
                    help='whether to save logs (False for debug & test, default: False)')
parser.add_argument('--multistep', type=str, default='',
                    help='learning_rate annealing schedule for multiple rounds e.g. 100,300, (default: )')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

# main
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

#TODO uncomment for debugging
# os.environ["WANDB_MODE"] = "offline"

if mpi_rank == 0:
    wandb.init(
        project=f"hetero_fedavg-{args.data}",
        notes=f"using contrastive loss, models:{args.models}",
        tags=[f"{args.data}", "CL"],
        config=args
    )
    wandb.run.name = args.runfile
    args = wandb.config

SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

models = args.models.split(',')
models *= mpi_size
# models = models[:mpi_size]

def evaluate(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    epoch_loss = 0
    correct = 0
    n_data = 0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels), in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        features = model(images)
        y_hat = classifier(features)
        # if y_pred.__class__.__name__ == 'GoogLeNetOutputs':
            # y_pred = y_pred.logits
        loss = criterion(y_hat, labels)
        epoch_loss += loss.item()

        # calculate train accuracy
        _, predicted = torch.max(y_hat, 1)
        correct += (predicted == labels).sum().item()
        n_data += len(labels)
    
    batch_idx += 1
    return epoch_loss / batch_idx, correct / n_data


def train(args, model, classifier, device, train_dataloader, csr, mu, client_id):
    """train one model"""
    model.train()
    classifier.train()
    initial_classifier = copy.deepcopy(classifier)
    n_iter = 0
    cont_loss_fn = SupConLoss(temperature=100.0, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    
    for epoch_idx in range(args.local_epochs):
        epoch_loss = 0
        epoch_cont_loss = 0
        correct = 0
        n_data = 0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = torch.cat([images[0], images[1]], dim=0)
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            features = model(images)
            if features.__class__.__name__ == 'GoogLeNetOutputs':
                features = features.logits
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            cont_loss = cont_loss_fn(features, labels) #supcontrast
            loss = cont_loss

            y_hat = classifier(f1)
            cls_loss = criterion(y_hat, labels) #classification loss
            loss += cls_loss

            # FedProx ; compute proximal loss
            prox_loss_fn = torch.nn.MSELoss()
            if batch_idx != 0:
                prox = torch.tensor(0.).to(device)
                for name, param in classifier.named_parameters():
                    initial_weight = initial_classifier.state_dict()[name]
                    prox += prox_loss_fn(param, initial_weight)
                if prox != 0:
                    loss += (prox * mu / 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate train accuracy
            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()
            epoch_cont_loss += cont_loss.item()
            
            # if batch_idx % args.log_interval == 0:
            # if batch_idx == 0 or batch_idx == (len(train_loader.dataset) / train_loader.batch_size):
                # print('[ROUND {} (RANK {}) CLIENT {}]'.format(csr, mpi_rank, client_id), ' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch_idx, batch_idx * train_loader.batch_size + len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            n_data += len(labels)

        train_loss = epoch_loss / (batch_idx+1)
        train_acc = correct / n_data
        print('[ROUND {} (RANK {}) CLIENT {}]'.format(csr, mpi_rank, client_id), 'CLoss/train:{:.3f}'.format(cont_loss))
        print('[ROUND {} (RANK {}) CLIENT {}]'.format(csr, mpi_rank, client_id), 'Loss/train:{:.3f}'.format(train_loss))
        print('[ROUND {} (RANK {}) CLIENT {}]'.format(csr, mpi_rank, client_id), 'Accuracy/train:{:.3f}'.format(train_acc))

        # val_loss, val_acc = evaluate(model, val_dataloader, device)
        # print('[ROUND {} (RANK {}) CLIENT {}]'.format(csr, mpi_rank, client_id), 'Loss/val:{:.3f}'.format(val_loss))
        # print('[ROUND {} (RANK {}) CLIENT {}]'.format(csr, mpi_rank, client_id), 'Accuracy/val:{:.3f}'.format(val_acc))

def move_state_dict(source, device):
    for param_tensor in source:
        source[param_tensor] = source[param_tensor].to(device)

def clone_state_dict(source):
    dest = {}
    for param_tensor in source:
        dest[param_tensor] = source[param_tensor].clone()
    return dest

def zero_state_dict_like(source):
    dest = {}
    for param_tensor in source:
        dest[param_tensor] = torch.zeros_like(source[param_tensor])
    return dest

# count number of participating nodes
num_virtual_nodes = args.virtual_per_node
n_parties = args.virtual_per_node * mpi_size
num_round_nodes = round(args.c * n_parties)
virtual_nodes = np.arange(args.virtual_per_node)

if n_parties == 0:
    print('exit program coz num_nodes == 0')

if mpi_rank == 0:
    print('number of participating nodes for each csr:', num_round_nodes)
    
if mpi_rank == 0: 
    print('Start loading data')
    start = time.time()

# load dataset
_, test_dataset, val_dataset, num_classes = utils.load_dataset(args.data, args.target_class_label_idx)
train_dataset = supcon_utils.load_dataset(args.data)

# collect y_test for test dataset balancing
y_test = np.array(test_dataset.targets)
test_indices_k = []
for k in range(num_classes):
    idx_k = np.where(y_test == k)[0]
    test_indices_k.append(idx_k)

try:
    configs = train_dataset.args
except:
    configs = {'learning_rate': args.learning_rate, 'batch_size': args.batch_size}

y_train = np.array(train_dataset.targets)
if mpi_rank == 0: 
    print(f'Done loading data (elapsed time: {time.time() - start} sec)')

# define runfile names
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
if args.runfile:
    runfile = args.runfile
else:
    runfile= '{}'.format(args.data)

if not args.output_path:
    output_path =  'results/debug'
else:
    output_path = args.output_path


learning_rate = configs['learning_rate']
batch_size = configs['batch_size']
hparam_dict = {"C": args.c, "B": batch_size, "E": args.local_epochs, "MU": args.mu}
if mpi_rank == 0:
    print(hparam_dict)

device = torch.device('cuda:{}'.format(args.gpu_id)) 
global_model, global_classifier = supcon_utils.define_model(modelname=models[0], num_classes=num_classes, device=device, dataname=args.data)

# define all client models in advance
client_models = []
client_classifiers = []
for client_id in range(n_parties): # models including global
    client_model, client_classifier = supcon_utils.define_model(modelname=models[client_id % mpi_size], num_classes=num_classes, device=device, dataname=args.data)
    client_models.append(client_model)
    client_classifiers.append(client_classifier)


val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)


global_state_dict = global_model.state_dict().copy()
global_cls_state_dict = global_classifier.state_dict().copy()
# initial global weight
if mpi_rank == 0:
    if 'sweep' not in output_path:
        torch.save(global_model, os.path.join(output_path, f'round0_global.pt'))
        torch.save(global_classifier, os.path.join(output_path, f'round0_global_cls.pt'))

# partition train data
net_index_map = partition_data(option=args.data_partition, y_train=y_train, n_parties=n_parties, beta=args.beta, num_labels=num_classes)
with open(os.path.join(output_path, 'net_index_map.txt'), 'w') as f:
    for client_id in range(n_parties):
        uniques, counts = np.unique(y_train[net_index_map[client_id]], return_counts=True)
        f.write(f'client{client_id},{dict(zip(uniques, counts))}\n')

val_loss, val_acc = 100, 0 
best_val_acc = 0
best_val_loss = np.inf
# early_stopping object
early_stopping = EarlyStopping(patience=100, verbose=True, path=os.path.join(output_path, 'best_model.pt'))
for round_idx in range(1, args.max_rounds + 1):
    # communication round
    # Step 1: sample clients
    active_per_round = round(args.c * n_parties)
    active_per_round_nodes = round(args.c * args.virtual_per_node)
    virtual_nodes = np.arange(args.virtual_per_node) + mpi_rank * args.virtual_per_node
    np.random.shuffle(virtual_nodes)
    # all_nodes = np.arange(n_parties)
    # np.random.shuffle(all_nodes)

    active_nodes = virtual_nodes[:active_per_round_nodes]
    # active_nodes = all_nodes[:active_per_round]
    # active_nodes = comm.bcast(active_nodes, root=0)
    
    print(f'[ROUND {round_idx}] active nodes:', active_nodes)
    
    # indices_per_rank = []
    # if active_per_round > mpi_size:
        # for i in np.arange(active_per_round):
            # if i % mpi_size == mpi_rank:
                # indices_per_rank.append(i)
# 
        # active_per_rank = active_nodes[indices_per_rank]
    # else:
        # active_per_rank = active_nodes[active_nodes % mpi_size == mpi_rank]

    # broadcast global model
    round_state_dict = comm.bcast(global_state_dict, root=0)
    round_cls_state_dict = comm.bcast(global_cls_state_dict, root=0)

    num_local_data = 0 
    sum_state_dict = zero_state_dict_like(global_state_dict) # to be local sum
    sum_cls_state_dict = zero_state_dict_like(global_cls_state_dict) # to be local sum

    # for active_idx, active in enumerate(active_per_rank):
    for active_idx, active in enumerate(active_nodes):
        # client id of virutal client
        client_id = active

        idx_tf = net_index_map[client_id]

        client_train_dataset = torch.utils.data.Subset(train_dataset, idx_tf)

        train_loader = torch.utils.data.DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

        # define model
        model = client_models[client_id].to(device)
        classifier = client_classifiers[client_id].to(device)

        # initialize model weights with global state dict
        model.load_state_dict(round_state_dict)
        classifier.load_state_dict(round_cls_state_dict)
        cur_entropy = train(args, model, classifier, device, train_loader, round_idx, mu=args.mu, client_id=client_id)

        # if round_idx % 10 == 0:
            # count unique labels at the active client
            # uniques, counts = np.unique(y_train[idx_tf], return_counts=True)
            # print(f'[ROUND{round_idx} (RANK {mpi_rank}) CLIENT {client_id}] datasize: {len(client_train_dataset)} labels:{dict(zip(uniques, counts))}')

        # aggregate virtual weights
        num_virtual_data = len(client_train_dataset)
        num_local_data += num_virtual_data

        for param_tensor in model.state_dict():
            sum_state_dict[param_tensor] += model.state_dict()[param_tensor].clone() * num_virtual_data

        for param_tensor in classifier.state_dict():
            sum_cls_state_dict[param_tensor] += classifier.state_dict()[param_tensor].clone() * num_virtual_data
        
        client_models[client_id] = model
        client_classifiers[client_id] = classifier

    # weight aggregation
    for param_tensor in sum_state_dict:
        param_buf = comm.allreduce(sum_state_dict[param_tensor], op=MPI.SUM)
        num_data = comm.allreduce(num_local_data, op=MPI.SUM)
        global_state_dict[param_tensor] = param_buf / num_data

    for param_tensor in sum_cls_state_dict:
        param_buf = comm.allreduce(sum_cls_state_dict[param_tensor], op=MPI.SUM)
        num_data = comm.allreduce(num_local_data, op=MPI.SUM)
        global_cls_state_dict[param_tensor] = param_buf / num_data
   
    ############ evaluate round result
    # average client test accuracy
    round_test_acc = 0
    round_test_loss = 0
    clients_to_test = np.where(np.arange(n_parties) % mpi_size == mpi_rank)[0]
    # for active_idx, active in enumerate(clients_to_test):
    for active_idx, active in enumerate(virtual_nodes):
        # client id of virutal client
        client_id = active

        model = client_models[client_id].to(device)
        classifier = client_classifiers[client_id].to(device)

        #  balance test dataset
        uniques, counts = np.unique(y_train[net_index_map[client_id]], return_counts=True)
        if args.data == 'emnist':
            counts = np.append(counts, 0)
            uniques = np.append(uniques, 0)
        datasize = np.sum(counts)
        label_dist = np.array([0] * num_classes)
        for j, l in enumerate(uniques):
            label_dist[l] = counts[j]
        label_dist = label_dist / datasize
        test_size = datasize / 3

        # partition test data
        sampled_test_indices = []
        for k, p in enumerate(label_dist):
            sampled_idx_k = np.random.choice(test_indices_k[k], int(test_size * p))
            sampled_test_indices.extend(sampled_idx_k)

        client_test_dataset = torch.utils.data.Subset(test_dataset, sampled_test_indices)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

        val_loss, val_acc = evaluate(model, classifier, client_test_loader, device)
        round_test_acc += val_acc
        round_test_loss += val_loss

        print('[ROUND {} (RANK {}) CLIENT {}]'.format(round_idx, mpi_rank, client_id), 'Loss/val:{:.3f}'.format(val_loss))
        print('[ROUND {} (RANK {}) CLIENT {}]'.format(round_idx, mpi_rank, client_id), 'Accuracy/val:{:.3f}'.format(val_acc))

        if round_idx % 100 == 0:
            # count unique labels at the active client
            if 'sweep' not in output_path:
                torch.save(model, os.path.join(output_path, f'round{round_idx}_client{client_id}.pt'))
                torch.save(classifier, os.path.join(output_path, f'round{round_idx}_client{client_id}_classifier.pt'))

    round_test_acc = comm.allreduce(round_test_acc, op=MPI.SUM) / n_parties 
    round_test_loss = comm.allreduce(round_test_loss, op=MPI.SUM) / n_parties 

    if mpi_rank == 0 :
        wandb.log({'global/loss': round_test_loss, 'global/acc': round_test_acc}, step=round_idx)
        print(f'[ROUND {round_idx}] Average test accuracy: {round_test_acc:.4f} (loss: {round_test_loss:.4f})')

        if best_val_loss > round_test_loss:
            if 'sweep' not in output_path:
                torch.save(model, os.path.join(output_path, 'best_model.pt'))
                torch.save(classifier, os.path.join(output_path, 'best_classifier.pt'))
            best_val_acc = round_test_acc
            best_val_loss = round_test_loss

        wandb.log({'best_global/loss': best_val_loss, 'best_global/acc': best_val_acc}, step=round_idx)

        print('[ROUND {}] Global validation accuracy:{:.4f}, loss:{:.4f} (BEST {:.4f}, {:.4f})'.format(round_idx, round_test_acc, round_test_loss, best_val_acc, best_val_loss))

        if round_test_acc >= 1:
            print('Early stopping at val_acc >= 1')
            break

        early_stopping(round_test_loss, model)

    # post average accuracy; sync client fc weights
    round_test_acc = 0
    round_test_loss = 0
    # for active_idx, active in enumerate(clients_to_test):
    for active_idx, active in enumerate(virtual_nodes):
        # client id of virutal client
        client_id = active

        model = client_models[client_id].to(device)
        classifier = client_classifiers[client_id].to(device)

        model.load_state_dict(global_state_dict)
        classifier.load_state_dict(global_cls_state_dict)

        val_loss, val_acc = evaluate(model, classifier, client_test_loader, device)
        round_test_loss += val_loss
        round_test_acc += val_acc

        print('[ROUND {} (RANK {}) CLIENT {}]'.format(round_idx, mpi_rank, client_id), 'Loss/post_val:{:.3f}'.format(val_loss))
        print('[ROUND {} (RANK {}) CLIENT {}]'.format(round_idx, mpi_rank, client_id), 'Accuracy/post_val:{:.3f}'.format(val_acc))

    round_test_acc = comm.allreduce(round_test_acc, op=MPI.SUM) / n_parties 
    round_test_loss = comm.allreduce(round_test_loss, op=MPI.SUM) / n_parties 
    if mpi_rank == 0 :
        wandb.log({'global_post/loss': round_test_loss, 'global_post/acc': round_test_acc}, step=round_idx)
        print(f'[ROUND {round_idx}] Average (POST) test accuracy: {round_test_acc:.4f} (loss: {round_test_loss:.4f})')

    #TODO commented for sweep
    if early_stopping.early_stop:
            print("Early stopping")
            break

if mpi_rank == 0:
    model = torch.load(os.path.join(output_path, 'best_model.pt')).to(device)
    classifier = torch.load(os.path.join(output_path, 'best_classifier.pt')).to(device)

    _, test_acc = evaluate(model, classifier, test_dataloader, device)
    print('Final test accuracy {:.3f}'.format(test_acc))
    wandb.log({'test_accuracy': test_acc})
    wandb.finish()