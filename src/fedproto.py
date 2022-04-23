import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import numpy as np
import utils
import wandb
from tqdm import tqdm
from utils import EarlyStopping

def update_weights_het(args, idx, global_protos, model, global_round=round):
    # Set mode to train model
    model.train()
    epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

    # Set optimizer for the local updates
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                        weight_decay=1e-4)

    for iter in range(args.train_ep):
        batch_loss = {'total':[],'1':[], '2':[], '3':[]}
        agg_protos_label = {}
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images, labels = images.to(self.device), label_g.to(self.device)

            # loss1: cross-entrophy loss, loss2: proto distance loss
            model.zero_grad()
            log_probs, protos = model(images)
            loss1 = self.criterion(log_probs, labels)

            loss_mse = nn.MSELoss()
            if len(global_protos) == 0:
                loss2 = 0*loss1
            else:
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)

            loss = loss1 + loss2 * args.ld
            loss.backward()
            optimizer.step()

            for i in range(len(labels)):
                if label_g[i].item() in agg_protos_label:
                    agg_protos_label[label_g[i].item()].append(protos[i,:])
                else:
                    agg_protos_label[label_g[i].item()] = [protos[i,:]]

            log_probs = log_probs[:, 0:args.num_classes]
            _, y_hat = log_probs.max(1)
            acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

            if self.args.verbose and (batch_idx % 10 == 0):
                print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                    global_round, idx, iter, batch_idx * len(images),
                    len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader),
                    loss.item(),
                    acc_val.item()))
            batch_loss['total'].append(loss.item())
            batch_loss['1'].append(loss1.item())
            batch_loss['2'].append(loss2.item())
        epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
        epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
        epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

    epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
    epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
    epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

    return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

def FedProto_modelheter(args, client_models, client_train_dataset, client_test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    global_protos = []
    idxs_users = np.arange(args.n)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = client_models[idx]
            #TODO
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global protos
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))

def main(args):
    wandb.init(
        project=f"KTpFL-{args.data}",
        notes=f"models:{args.models}",
        tags=[f"{args.data}"],
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

    '''gpu'''
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    '''Import Data'''
    train_dataset, test_dataset, val_dataset, num_classes = utils.load_dataset(
        data_split=args.data, target_label_idx=0)
    public_dataset = load_public_dataset(args.data, datasize=args.public_datasize)
    y_train = np.array(train_dataset.targets)
    net_index_map = utils.partition_data(option=args.data_partition, num_labels=num_classes, y_train=y_train, n_parties=args.n_parties)
    y_test = np.array(test_dataset.targets)

    # get client test
    test_indices_k = []
    for k in range(num_classes):
        idx_k = np.where(y_test == k)[0]
        test_indices_k.append(idx_k)

    '''Create client models'''
    print('Creating client models and datasets')
    client_models = []
    client_train_datasets = []
    client_test_datasets = []
    for client_id in range(args.n_parties):  # models including global
        # create client model
        client_model = utils.define_model(
            modelname=models[client_id % len(np.unique(models))], num_classes=num_classes, device=device, dataname=args.data)
        client_models.append(client_model)

        client_train_dataset = torch.utils.data.Subset(train_dataset, net_index_map[client_id])#  balance test dataset
        client_train_datasets.append(client_train_dataset)

        # sample test dataset
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

        sampled_test_indices = []
        for k, p in enumerate(label_dist):
            sampled_idx_k = np.random.choice(test_indices_k[k], int(test_size * p))
            sampled_test_indices.extend(sampled_idx_k)

        client_test_dataset = torch.utils.data.Subset(test_dataset, sampled_test_indices)
        client_test_datasets.append(client_test_dataset)
    print('DONE creating client models and datasets')

    FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.04,
                        help='the fraction of clients: C')
    parser.add_argument('--train_ep', type=int, default=1,
                        help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=4,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--alg', type=str, default='fedproto', help="algorithms")
    parser.add_argument('--mode', type=str, default='task_heter', help="mode")
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--data_dir', type=str, default='../data/', help="directory of dataset")
    parser.add_argument('--data', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")

    # Local arguments
    parser.add_argument('--ways', type=int, default=3, help="num of classes")
    parser.add_argument('--shots', type=int, default=100, help="num of shots")
    parser.add_argument('--train_shots_max', type=int, default=110, help="num of shots")
    parser.add_argument('--test_shots', type=int, default=15, help="num of shots")
    parser.add_argument('--stdev', type=int, default=2, help="stdev of ways")
    parser.add_argument('--ld', type=float, default=1, help="weight of proto loss")
    parser.add_argument('--ft_round', type=int, default=10, help="round of fine tuning")
    args = parser.parse_args()
    main(args)
