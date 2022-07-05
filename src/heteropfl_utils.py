import copy
from losses import SupConLoss
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

def args_parser():
    ### Args
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--target_class_label_idx', type=int, default=0, help='for navon')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--output_path', type=str, default=None, help="path to write output files")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--runfile', type=str, default=None)

    ### FL-related paprams ###
    parser.add_argument('--c', type=float, default=1,
                        help='Fraction of participating clients at each communication round (default: 0.1)')
    parser.add_argument('--max_rounds', type=int, default=200)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--virtual_per_node', type=int, default=2,
                        help='how many virtual nodes to iterate in each mpi process (default: 1)')
    parser.add_argument('--data_partition', type=str, default='noniid-twoclass', help='iid|noniid-labelskew|noniid-labeldir|noniid-twoclass')
    parser.add_argument('--beta', type=float, default=0.5, help='The concentration parameter of the Dirichlet distribution for heterogeneous partition.')
    # hetero
    parser.add_argument('--models', type=str, default='googlenet')

    # parser.add_argument('--print-weight', type=bool, default=False,
                        # help='Whether to print weight and gradients (for debug, default: False)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Client update epochs at every communication round (default: 1)')
    parser.add_argument('--mu', type=float, default=0,
                        help='Hyperparameter for Proximal term. FedAvg if 0. (default: 0)')
    parser.add_argument('--is_experiment', type=bool, default=False,
                        help='whether to save logs (False for debug & test, default: False)')
    parser.add_argument('--multistep', type=str, default='',
                        help='lr annealing schedule for multiple rounds e.g. 100,300, (default: )')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--tsne', action='store_true')
    return parser.parse_args()

def train_cl(model, classifier, device, train_dataloader, mu, configs, local_epochs=1):
    """train one model using CL"""
    model.train()
    classifier.train()
    initial_classifier = copy.deepcopy(classifier)
    cont_loss_fn = SupConLoss(temperature=100.0, device=device)
    # cont_loss_fn = SupConLoss(temperature=0.07)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=configs['learning_rate'], betas=(0.5, 0.999))
    
    for epoch_idx in range(local_epochs):
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
            # features = torch.nn.normalize(features, dim=1)
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
            
            n_data += len(labels)

        train_closs = epoch_cont_loss / len(train_dataloader)
        train_loss = epoch_loss / len(train_dataloader)
        train_acc = correct / n_data
        return train_closs, train_loss, train_acc

def evaluate_cl(model, classifier, loader, device):
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
        loss = criterion(y_hat, labels)
        epoch_loss += loss.item()

        # calculate train accuracy
        _, predicted = torch.max(y_hat, 1)
        correct += (predicted == labels).sum().item()
        n_data += len(labels)
    
    batch_idx += 1
    return epoch_loss / len(loader), correct / n_data

def train(model, device, train_dataloader, mu, configs, local_epochs=1):
    """train one model using CL"""
    model.train()
    initial_model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configs['learning_rate'], betas=(0.5, 0.999))
    
    for epoch_idx in range(local_epochs):
        epoch_loss = 0
        epoch_cont_loss = 0
        correct = 0
        n_data = 0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            y_hat = model(images)
            if y_hat.__class__.__name__ == 'GoogLeNetOutputs':
                y_hat = y_hat.logits
            loss = criterion(y_hat, labels) #classification loss

            # FedProx ; compute proximal loss
            prox_loss_fn = torch.nn.MSELoss()
            if batch_idx != 0:
                prox = torch.tensor(0.).to(device)
                for name, param in model.named_parameters():
                    if 'fc' in name:
                        initial_weight = initial_model.state_dict()[name]
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
            
            n_data += len(labels)

        train_loss = epoch_loss / len(train_dataloader)
        train_acc = correct / n_data
        return train_loss, train_acc

def evaluate(model, loader, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    n_data = 0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels), in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        y_hat = model(images)
        loss = criterion(y_hat, labels)
        epoch_loss += loss.item()

        # calculate train accuracy
        _, predicted = torch.max(y_hat, 1)
        correct += (predicted == labels).sum().item()
        n_data += len(labels)
    
    batch_idx += 1
    return epoch_loss / len(loader), correct / n_data