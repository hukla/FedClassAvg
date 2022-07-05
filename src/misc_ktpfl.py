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

#　Iterate over data classes
class data(torch.utils.data.Dataset):
    def __init__(self,X, y):
        self.X = np.array(X[:len(y)]) #due to drop last
        self.y = np.array(y)
    def __getitem__(self, item):
        X = self.X[item]
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0).astype(np.float32)
        else:
            X = np.transpose(X, (2,0,1)).astype(np.float32)
        return X, self.y[item]

    def __len__(self):
        return len(self.X)

def generate_partial_data(dataset, classes, datasize=5000):
    targets = dataset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    data_indices = []
    for c in classes:
        idx_c = list(np.where(targets == c)[0])
        data_indices.extend(idx_c)
    data_indices = np.random.choice(data_indices, size=datasize, replace=False)
    partial_dataset = copy.deepcopy(dataset)
    partial_dataset.data = partial_dataset.data[data_indices]
    if isinstance(partial_dataset.targets, list):
        partial_dataset.targets = np.array(partial_dataset.targets)

    partial_dataset.targets = partial_dataset.targets[data_indices]
    return partial_dataset


def load_public_dataset(dataname, datasize=5000):
    if 'cifar' in dataname:
        # load cifar100 0: apple, 2: baby, 20: chair, 63:porcupine, 71: sea, 82: sunflower
        # cifar10 0: airplane, 1: automobile, 3: bird, 4: cat, 5: deer, 6:dog, 7: frog, 8: horse, 9: ship, 10: truck
        public_classes = [0, 2, 20, 63, 71, 82]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, ],
                                         std=[0.229, 0.224, 0.225])

        trsfm_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

        trsfm_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR100(
            './data', train=True, download=True, transform=trsfm_train)

        public_dataset = generate_partial_data(train_dataset, public_classes, datasize)

    elif 'mnist' in dataname:
        # "public_classes": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        public_dataset = datasets.MNIST(root='./data',
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)

        public_dataset = generate_partial_data(public_dataset, np.arange(10), datasize)

    return public_dataset

def distill_one_model(model, max_epochs, device, train_dataloader, optimizer, criterion, client_id, csr=-1):
    model.train()
    n_data = 0

    for epoch_idx in range(max_epochs):
        epoch_loss = 0
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data, labels = data.to(device), labels.to(device)
            target_label = labels
            n_data += len(labels)

            optimizer.zero_grad()
            y_hat = model(data)
            if y_hat.__class__.__name__ == 'GoogLeNetOutputs':
                y_hat = y_hat.logits
            y_hat = torch.nn.functional.softmax(y_hat, dim=1) # distillation target is softmax
            loss = criterion(y_hat, target_label)

            loss.backward()
            optimizer.step()

            # calculate train accuracy
            epoch_loss += loss.item() * len(labels)
            del data, labels, y_hat, loss, predicted
            
        train_loss = epoch_loss / (batch_idx + 1) 
    
    optimizer.zero_grad(set_to_none=True)

    return train_loss

def train_one_model(model, max_epochs, device, train_dataloader, optimizer, criterion, client_id, csr=-1):
    model.train()
    n_iter = 0
    for epoch_idx in range(max_epochs):
        epoch_loss = 0
        correct = 0
        n_data = 0
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data, labels = data.to(device), labels.to(device)
            n_data += len(labels)

            optimizer.zero_grad()
            y_hat = model(data)
            if y_hat.__class__.__name__ == 'GoogLeNetOutputs':
                y_hat = y_hat.logits
            loss = criterion(y_hat, labels)

            loss.backward()
            optimizer.step()

            # calculate train accuracy
            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item() * len(labels)
            
            n_iter += 1
            del data, labels, y_hat, loss, predicted


        train_loss = epoch_loss / (batch_idx + 1)
        train_acc = correct / n_data

    optimizer.zero_grad(set_to_none=True)

    return train_loss, train_acc

def evaluate_one_model(model, loader, criterion, device):
    """ evluate one model

    Args:
        model (model): model to evaluate
        loader (torch.utils.data.Dataloader): data to evaluate
        criterion (torch.nn.CrossEntropy()): loss function
        device (str): device

    Returns:
        loss (float): evaluation loss
        accuracy (float): evaluation accuracy
    """    
    model.eval()
    epoch_loss = 0
    correct = 0
    n_data = 0
    batch_idx = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            n_data += len(labels)
            data, labels = data.to(device), labels.to(device)

            y_pred = model(data)
            if y_pred.__class__.__name__ == 'GoogLeNetOutputs':
                y_pred = y_pred.logits
            loss = criterion(y_pred, labels)
            epoch_loss += loss.item() * y_pred.shape[0]

            # calculate train accuracy
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == labels).sum().item()
            del data, labels, y_pred, loss, predicted

    return epoch_loss / (batch_idx+1) , correct / n_data

def predict(model, dataloader, device, T):
    model.eval()
    out= []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            logit = model(images)

            Tsoftmax = nn.Softmax(dim=1)
            # Add temperature coefficient T
            output_logit = Tsoftmax(logit.float()/T)

            out.append(output_logit.cpu().numpy())
            del images, labels, logit, output_logit
    out = np.concatenate(out)
    return out


