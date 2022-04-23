import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from collections import OrderedDict
import models 

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def load_dataset(data_split, target_label_idx=0):
    
    # construct data loader
    if data_split == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif data_split == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32
    elif 'mnist' in data_split:
        mean = (0.5, )
        std = (0.5, )
        size = 28
    else:
        raise ValueError('dataset not supported: {}'.format(data_split))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if data_split == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data',
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        train_dataset.configs = {'learning_rate': 0.0001, 'batch_size': 64}
    elif data_split == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data',
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif data_split == 'emnist':
        train_dataset = torchvision.datasets.EMNIST('./data', split='letters', train=True, download=True, transform=TwoCropTransform(train_transform))
        train_dataset.configs = {'learning_rate': 0.0005, 'batch_size': 64}
    elif data_split == 'fashion-mnist':
        train_dataset = torchvision.datasets.FashionMNIST('./data', train=True, download=True,
            transform=TwoCropTransform(train_transform))
        train_dataset.configs = {'learning_rate': 0.0006, 'batch_size': 64}
    elif data_split == 'path':
        train_dataset = datasets.ImageFolder(root='./data',
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(data_split)

    return train_dataset


def define_model(modelname, num_classes, device, dataname, feat_dim=512):
    if modelname == 'resnet':
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(OrderedDict([
            ('fcin', nn.Linear(512, feat_dim)),
        ]))
        if 'mnist' in dataname:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif modelname == 'shufflenet':
        model = torchvision.models.shufflenet_v2_x1_0()
        model.fc = nn.Sequential(OrderedDict([
            ('fcin', nn.Linear(1024, feat_dim)),
        ]))
        if 'mnist' in dataname:
            model.conv1[0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif modelname == 'googlenet':
        model = torchvision.models.googlenet(init_weights=True)
        model.fc = nn.Sequential(OrderedDict([
            ('fcin', nn.Linear(1024, feat_dim)),
        ])).to(device)
        if 'mnist' in dataname:
            model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif modelname == 'alexnet':
        if 'imagenet' in dataname:
            model = torchvision.models.alexnet()
            model.classifier = nn.Sequential(nn.Dropout())
            model.fc = nn.Sequential(OrderedDict([
                ('fcin', nn.Linear(256 * 256 * 3, feat_dim)),
            ]))
        elif 'mnist' in dataname:
            model = models.alexnet_mnist(1, feat_dim)
        else:
            model = models.alexnet_mnist(3, feat_dim)

    model = model.to(device)
    
    classifier = nn.Sequential(OrderedDict([
        ('relu', nn.ReLU()),
        ('fcout', nn.Linear(feat_dim, num_classes))
    ])).to(device)
        
    return model, classifier

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr