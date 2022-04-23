import json
import copy
import math
import os
import random
import string
import time
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from mpi4py import MPI
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import heteropfl_utils
import utils
from utils import EarlyStopping, define_model, load_dataset, partition_data
from tqdm import tqdm

### Args
args = heteropfl_utils.args_parser()

# load data configs
with open(f'configs/{args.data}.json', 'r') as config_json:
    configs = json.load(config_json)

args.learning_rate = configs['learning_rate']
args.batch_size = configs['batch_size']

# main
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()
device = torch.device('cuda:{}'.format(args.gpu_id))

if mpi_rank == 0:
    wandb.init(
        project=f"hetero_fedavg-{args.data}",
        notes=f"models:{args.models}",
        tags=[f'{args.data}'],
        config=args,
    )
    args = wandb.config
    wandb.run.name = args.runfile

SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

models = args.models.split(',')
models *= mpi_size

# count number of participating nodes
n_parties = args.virtual_per_node * mpi_size
num_round_nodes = round(args.c * n_parties)
virtual_nodes = np.arange(args.virtual_per_node) + mpi_rank * args.virtual_per_node # virtual nodes for mpi rank

if n_parties == 0:
    print('exit program coz num_nodes == 0')

if mpi_rank == 0:
    print('number of participating nodes for each csr:', num_round_nodes)

if mpi_rank == 0:
    print('Start loading data')
    start = time.time()

# load dataset
train_dataset, test_dataset, val_dataset, num_classes = load_dataset(args.data, args.target_class_label_idx)

if mpi_rank == 0:
    print(f'Done loading data (elapsed time: {time.time() - start} sec)')

y_train = np.array(train_dataset.targets)
y_test = np.array(test_dataset.targets)

# collect y_test for test dataset balancing
test_indices_k = []
for k in range(num_classes):
    idx_k = np.where(y_test == k)[0]
    test_indices_k.append(idx_k)

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

if mpi_rank == 0 and not os.path.exists(os.path.join(output_path, 'best')):
    os.mkdir(os.path.join(output_path, 'best'))


#define all client models in advance
global_model = define_model(modelname=models[0], num_classes=num_classes, device=device, dataname=args.data)
client_models = []
for client_id in range(n_parties): # models including global
    client_model = define_model(modelname=models[client_id % mpi_size], num_classes=num_classes, device=device, dataname=args.data)
    client_models.append(client_model)

val_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

global_state_dict = global_model.state_dict().copy()
if mpi_rank == 0:
    if 'sweep' not in output_path:
        # initial global weight
        torch.save(global_model, os.path.join(output_path, f'round0_global.pt'))

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
early_stopping = EarlyStopping(patience=100, verbose=True)

pretrain = True
if pretrain:
    # pretrain nodes
    for virtual_node in virtual_nodes:
        # client id of virutal client
            client_id = virtual_node

            idx_tf = net_index_map[client_id]

            client_train_dataset = torch.utils.data.Subset(train_dataset, idx_tf)

            train_loader = torch.utils.data.DataLoader(client_train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=4, drop_last=True)

            # define model
            model = client_models[client_id].to(device)

            train_loss, train_acc = heteropfl_utils.train(model=model, device=device,
                                                                        train_dataloader=train_loader, mu=args.mu, configs=configs,
                                                                        local_epochs=args.local_epochs)
            print('[ROUND {} (RANK {}) CLIENT {}]'.format('init', mpi_rank, client_id), 'Loss/train:{:.3f}'.format(train_loss))
            print('[ROUND {} (RANK {}) CLIENT {}]'.format('init', mpi_rank, client_id), 'Accuracy/train:{:.3f}'.format(train_acc))

            client_models[client_id] = model

for round_idx in range(1, args.max_rounds + 1):
    best_round = False
    # communication round
    # Step 1: sample clients
    np.random.shuffle(virtual_nodes)
    active_nodes = virtual_nodes[:num_round_nodes]

    if mpi_rank == 0:
        print(f'[ROUND {round_idx}] active nodes:', active_nodes)

    for active_idx, active in enumerate(virtual_nodes):
        # client id of virutal client
        client_id = active

        # print(f'CLIENT {client_id} model: {models[client_id % mpi_size]}')
        idx_tf = net_index_map[client_id]
        client_train_dataset = torch.utils.data.Subset(train_dataset, idx_tf)
        train_loader = torch.utils.data.DataLoader(client_train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=4, drop_last=True)

        model = client_models[client_id].to(device)

        train_loss, train_acc = heteropfl_utils.train(model=model, device=device,
                                                    train_dataloader=train_loader, mu=args.mu, configs=configs,
                                                    local_epochs=args.local_epochs)
        print('[ROUND {} (RANK {}) CLIENT {}]'.format(round_idx, mpi_rank, client_id), 'Loss/train:{:.3f}'.format(train_loss))
        print('[ROUND {} (RANK {}) CLIENT {}]'.format(round_idx, mpi_rank, client_id), 'Accuracy/train:{:.3f}'.format(train_acc))

        client_models[client_id] = model

    ############ evaluate round result
    # average client test accuracy
    round_test_acc = 0
    round_test_loss = 0

    for active_idx, active in enumerate(virtual_nodes):
        # client id of virutal client
        client_id = active
        idx_tf = net_index_map[client_id]

        model = client_models[client_id].to(device)

        local_test_indices = utils.get_local_test_indices(args.data, num_classes, y_train[idx_tf], test_indices_k)
        client_test_dataset = torch.utils.data.Subset(test_dataset, local_test_indices)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

        val_loss, val_acc = heteropfl_utils.evaluate(model, client_test_loader, device)
        round_test_acc += val_acc
        round_test_loss += val_loss

        print('[ROUND {} (RANK {}) CLIENT {}]'.format(round_idx, mpi_rank, client_id), 'Loss/val:{:.3f}'.format(val_loss))
        print('[ROUND {} (RANK {}) CLIENT {}]'.format(round_idx, mpi_rank, client_id), 'Accuracy/val:{:.3f}'.format(val_acc))

        if round_idx % 100 == 0 and 'sweep' not in output_path:
            # save model weights
            torch.save(model, os.path.join(output_path, f'round{round_idx}_client{client_id}.pt'))
            print(f'RANK {mpi_rank} saved model {client_id}')

    round_test_acc = comm.allreduce(round_test_acc, op=MPI.SUM) / n_parties
    round_test_loss = comm.allreduce(round_test_loss, op=MPI.SUM) / n_parties

    if mpi_rank == 0 :
        wandb.log({'global/loss': round_test_loss, 'global/acc': round_test_acc}, step=round_idx)
        print(f'[ROUND {round_idx}] Average test accuracy: {round_test_acc:.4f} (loss: {round_test_loss:.4f})')

        if best_val_loss > round_test_loss:
            torch.save(model, os.path.join(output_path, 'best_model.pt'))
            best_val_acc = round_test_acc
            best_val_loss = round_test_loss
            best_round = True

        wandb.log({'best_global/loss': best_val_loss, 'best_global/acc': best_val_acc}, step=round_idx)

        print('[ROUND {}] Global validation accuracy:{:.4f}, loss:{:.4f} (BEST {:.4f}, {:.4f})'.format(round_idx, round_test_acc, round_test_loss, best_val_acc, best_val_loss))

        if round_test_acc >= 1:
            print('Early stopping at val_acc >= 1')
            break

    best_val_loss = comm.bcast(best_val_loss, root=0)
    best_val_acc = comm.bcast(best_val_acc, root=0)
    best_round = comm.bcast(best_round, root=0)

    # save best model weights
    if best_round and 'sweep' not in output_path:
        for client_id in virtual_nodes:
            torch.save(client_models[client_id], os.path.join(output_path, f'best/model_client{client_id}.pt'))
            print(f'RANK {mpi_rank} saved model {client_id}')
        best_round = False

    if mpi_rank == 0 and args.tsne:
        # plot t-SNE
        test_dl = DataLoader(test_dataset, 1024, True)
        preds, label_arr, client_label, performance = utils.generate_tsne_features(client_models[:len(virtual_nodes)], client_classifiers, test_dl, device)
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
        act_tsne = tsne.fit_transform(preds.numpy())
        label_arr = label_arr.cpu()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        sns.scatterplot(x=act_tsne[:,0], y=act_tsne[:,1], hue=label_arr, palette=sns.color_palette("colorblind", len(np.unique(label_arr))), alpha=0.5, ax=axes[0], legend=False, markers='o')
        sns.scatterplot(x=act_tsne[:,0], y=act_tsne[:,1], hue=client_label, palette=sns.color_palette("hls", len(np.unique(client_label))), alpha=0.5, ax=axes[1], legend=False, markers='o')
        axes[0].set_facecolor('white')
        axes[1].set_facecolor('white')
        wandb.log({"tsne": fig})
        print(f'[ROUND {round_idx}] t-SNE plotted')

    early_stopping(round_test_loss, model)

    if early_stopping.early_stop:
            print("Early stopping")
            break

if mpi_rank == 0:
    print('Best test accuracy:', best_val_acc)
    wandb.finish()
