import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import models
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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

def partition_data(option='iid', min_require_size=10, num_labels=10, y_train=None, n_parties=100, beta=0.5):
    min_size = 0
    num_data = len(y_train)
    np.random.seed(2021)
    net_dataidx_map = {}

    if option == 'iid':
        shard_size = round(len(y_train) / n_parties)
        idx_batch = np.array_split(np.arange(len(y_train)), n_parties)
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif option == 'noniid-labelskew':
        shard_size = int(len(y_train) / (n_parties * 2))

        # Sort indices by class
        sorted_indices = []
        for k in range(num_labels):
            idx_k = np.where(y_train == k)[0]
            # np.random.shuffle(idx_k)
            sorted_indices.extend(idx_k.tolist())
        
        idx_shard = [sorted_indices[shard_size * i : shard_size * (i + 1)] for i in range(n_parties * 2)]

        for j in range(n_parties):
            idx_batch = idx_shard[2 * j] + idx_shard[2 * j + 1]
            np.random.shuffle(idx_batch)
            net_dataidx_map[j] = idx_batch

    elif option == 'noniid-twoclass':
        labels = np.arange(num_labels)
        shard_size = int(len(y_train) / (n_parties * 2))

        # Sort indices by class
        sorted_indices = []
        empty_labels = []
        for k in range(num_labels):
            idx_k = np.where(y_train == k)[0]
            # np.random.shuffle(idx_k)
            sorted_indices.append(idx_k.tolist())       
            if len(idx_k) == 0:
                empty_labels.append(k)
        labels = np.delete(labels, empty_labels)
        
        for j in range(n_parties):
            selected_labels = np.random.choice(labels, 2, replace=False)
            idx_shard = np.append(np.random.choice(sorted_indices[selected_labels[0]], shard_size), np.random.choice(sorted_indices[selected_labels[1]], shard_size))
            # idx_batch = idx_shard[2 * j] + idx_shard[2 * j + 1]
            np.random.shuffle(idx_shard)
            net_dataidx_map[j] = list(idx_shard)

    elif option == 'noniid-labeldir':
        # revised from https://github.com/Xtra-Computing/NIID-Bench/blob/main/utils.py
        min_size = 0
        num_data = y_train.shape[0]
        np.random.seed(2021)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(num_labels):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < num_data / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    else:
        raise NameError('invalid data partition option')

    return net_dataidx_map

def get_local_test_indices(data, num_classes, y_train, test_indices_k, test_ratio=.3):
    # balance test dataset for local train data distribution
    uniques, counts = np.unique(y_train, return_counts=True)
    if data == 'emnist':
        counts = np.append(counts, 0)
        uniques = np.append(uniques, 0)
    train_datasize = np.sum(counts)

    label_dist = np.array([0] * num_classes)
    for j, l in enumerate(uniques):
        label_dist[l] = counts[j]
    label_dist = label_dist / train_datasize
    test_datasize = train_datasize * test_ratio

    # partition test data
    sampled_test_indices = []
    for k, p in enumerate(label_dist):
        sampled_idx_k = np.random.choice(test_indices_k[k], int(test_datasize * p))
        sampled_test_indices.extend(sampled_idx_k)
    return sampled_test_indices

def load_dataset(data_split, target_label_idx):
    """ load dataset

    Args:
        data_split (str): dataname:splitnum
        target_label_idx (int): 0: shape, 1: texture

    Returns:
        train_dataset (torch.utils.data.Dataset): train dataset
        test_dataset (torch.utils.data.Dataset): test dataset
        val_dataset (torch.utils.data.Dataset): val dataset
        num_classes (int): # classes of dataset
    """    
    if len(data_split.split(':')) == 2:
        dataname = data_split.split(':')[0]
        split = data_split.split(':')[1]
    else:
        dataname = data_split.split(':')[0]

    if dataname == 'cifar10':
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

        train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=trsfm_train)
        test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=trsfm_test)
        val_dataset = test_dataset
        num_classes = 10
        train_dataset.configs = {'lr': 0.0001, 'batch_size': 64}

    if dataname == 'emnist':
        trfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
                ])
        train_dataset = torchvision.datasets.EMNIST('./data', split='letters', train=True, download=True, transform=trfm)
        test_dataset = torchvision.datasets.EMNIST('./data', split='letters', train=False, download=True, transform=trfm)
        val_dataset = test_dataset
        num_classes = len(train_dataset.classes) # 26 excluding N/A
        train_dataset.configs = {'lr': 0.0005, 'batch_size': 64}

    if dataname == 'fashion-mnist':
        trfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
                ])
        train_dataset = torchvision.datasets.FashionMNIST('./data', train=True, download=True,
            transform=trfm)
        test_dataset = torchvision.datasets.FashionMNIST('./data', train=False, download=True,
            transform=trfm)
        val_dataset = test_dataset
        num_classes = 10
        train_dataset.configs = {'lr': 0.0006, 'batch_size': 64}

    if dataname == 'mnist':
        trfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
                ])
        train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=trfm)
        test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=trfm)
        val_dataset = test_dataset
        num_classes = 10

    return train_dataset, test_dataset, val_dataset, num_classes

def define_model(modelname, num_classes, device, dataname):
    if modelname == 'resnet':
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(OrderedDict([
            ('fcin', nn.Linear(512, 512)),
            ('relu', nn.ReLU()),
            ('fcout', nn.Linear(512, num_classes)),
        ]))
        if 'mnist' in dataname:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif modelname == 'shufflenet':
        model = torchvision.models.shufflenet_v2_x1_0()
        model.fc = nn.Sequential(OrderedDict([
            ('fcin', nn.Linear(1024, 512)),
            ('relu', nn.ReLU()),
            ('fcout', nn.Linear(512, num_classes)),
        ]))
        if 'mnist' in dataname:
            model.conv1[0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif modelname == 'googlenet':
        model = torchvision.models.googlenet(init_weights=True)
        model.fc = nn.Sequential(OrderedDict([
            ('fcin', nn.Linear(1024, 512)),
            ('relu', nn.ReLU()),
            ('fcout', nn.Linear(512, num_classes)),
        ]))
        if 'mnist' in dataname:
            model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif modelname == 'alexnet':
        if 'imagenet' in dataname:
            model = torchvision.models.alexnet()
            model.classifier = nn.Sequential(nn.Dropout())
            model.fc = nn.Sequential(OrderedDict([
                ('fcin', nn.Linear(256 * 256 * 3, 512)),
                ('relu', nn.ReLU()),
                ('fcout', nn.Linear(512, num_classes)),
            ]))
        elif 'mnist' in dataname:
            model = models.alexnet_mnist(1, num_classes)
        else:
            model = models.alexnet_mnist(3, num_classes)
    model = model.to(device)
        
    return model

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path=None):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.path is not None:
                self.save_checkpoint(val_loss, model)
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'============EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.path is not None:
                self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'============Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.path is not None:
            torch.save(model, self.path)
        self.val_loss_min = val_loss

def generate_tsne_features(models, test_dl, device='cuda:0'):
    preds = torch.Tensor([])
    label_arr = torch.Tensor([])
    client_label = []
    performance = dict()

    for client_id in range(len(models)):
        # load model
        model = copy.deepcopy(models[client_id])
        model.to(device)
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                if output.__class__.__name__ == 'GoogLeNetOutputs':
                    output = output.logits
                activation[name] = output.detach()
            return hook

        model.fc.fcin.register_forward_hook(get_activation('fc.fcin'))
        for images, labels in test_dl:
            images = images.to(device)
            with torch.no_grad():
                pred = model(images)
                if pred.__class__.__name__ == 'GoogLeNetOutputs':
                    pred = pred.logits

                act = activation['fc.fcin']
            _, pred_labels = torch.max(pred, 1)

            correct = (pred_labels.cpu() == labels)
            act = act[correct]
            labels = labels[correct]

            preds = torch.cat([preds, act.cpu().data])
            label_arr = torch.cat([label_arr, labels.data])
            client_label.extend([client_id] * len(labels))
            performance[client_id] = correct.sum() / 1024

            break
    return preds, label_arr, client_label, performance

def generate_tsne_features_cl(models, classifiers, test_dl, device='cuda:0'):
    preds = torch.Tensor([])
    label_arr = torch.Tensor([])
    client_label = []
    performance = dict()
    for client_id in range(len(models)):
        # load model
        model = copy.deepcopy(models[client_id]).to(device)
        classifier = copy.deepcopy(classifiers[client_id]).to(device)
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                if output.__class__.__name__ == 'GoogLeNetOutputs':
                    output = output.logits
                activation[name] = output.detach()
            return hook
     
        model.fc.fcin.register_forward_hook(get_activation('fc.fcin'))
        for images, labels in test_dl:
            images = images.to(device)
            with torch.no_grad():
                features = model(images)
                if features.__class__.__name__ == 'GoogLeNetOutputs':
                    features = features.logits
                pred = classifier(features)

                act = activation['fc.fcin']
            _, pred_labels = torch.max(pred, 1)
            correct = (pred_labels.cpu() == labels)
            features = features[correct]
            act = act[correct]
            labels_ = labels[correct]

            preds = torch.cat([preds, act.cpu().data])
            label_arr = torch.cat([label_arr, labels_.data])
            client_label.extend([client_id] * len(labels_))
            performance[client_id] = correct.sum() / 1024

            break
    
    return preds, label_arr, client_label, performance

if __name__ == "__main__":
    # load dataset
    train_dataset, test_dataset, val_dataset, num_classes = load_dataset('cifar10', 0)
    y_train = np.array(train_dataset.targets)
    net_index_map = partition_data(option='noniid-labelskew', min_require_size=10, num_labels=10, y_train=y_train, n_parties=100, beta=0.5)
    print(net_index_map)