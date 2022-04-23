# https://github.com/jiaxue1993/pytorch-material-classification/blob/master/dataloader/gtos.py
#%%
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

def find_classes(classdir):
    classes = []
    class_to_idx = {}
    with open(classdir, 'r') as f:
        for line in f:
            label, name = line.split(' ')
            classes.append(name[:-1])
            class_to_idx[name] = int(label) - 1
    return classes, class_to_idx

class GTOSTransform():
    def __init__(self, image_size):
        self.data_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.447, 0.388, 0.340],
                                         std=[0.216, 0.204, 0.197]),
            ])
    
    def __call__(self, img):
        return self.data_transform(img)

class GTOSDataset(torch.utils.data.Dataset):
    def __init__(self, label_path, transforms=None):
        # find classes
        datapath  = '/'.join(label_path.split('/')[:-2])
        classes, class_to_idx = find_classes(os.path.join(datapath, 'labels/classInd.txt'))
        self.classes = classes
        self.label2idx = class_to_idx
        self.idx2label = np.array(self.classes)

        # make datset
        self.datapath = datapath

        self.file_list = list()
        self.targets = list()

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = GTOSTransform(224)

        with open(label_path, 'r') as lines:
            for line in lines:
                name, label = line.split(' ')
                name = name.split('/')[-1]
                for filename in os.listdir(os.path.join(self.datapath, 'color_imgs', name)):
                    img = os.path.join(self.datapath, 'color_imgs', name, filename)
                    assert os.path.isfile(img)
                    self.file_list.append(img)
                    self.targets.append(int(label) - 1)
        
        self.targets = np.array(self.targets)
        self.configs={'lr': 0.0003, 'batch_size': 128}

        # load labels
        # for img_path in self.file_list:
            # label = img_path.split('/')[0]
            # self.targets.append(label)
        
        # self.targets = np.array([self.label2idx[label] for label in self.labels])

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # load data
        img_path = self.file_list[index]
        label = self.targets[index]
    
        img = Image.open(img_path)
        img = self.transforms(img)
        if img.shape != torch.Size([3, 224, 224]):
            img = torch.cat([img, img, img], axis=0)

        return img, label

def compute_image_mean(image_loader):
    ####### COMPUTE MEAN / STD

    # placeholders
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs, label in tqdm(image_loader):
        psum    += inputs.sum(axis        = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

    ####### FINAL CALCULATIONS

    # pixel count
    count = len(dataset) * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

def plot_samples(image_loader):
    fig, axes = plt.subplots(nrows=1, ncols=5)
    for images, labels in image_loader:
        for idx, (image, label) in enumerate(zip(images, labels)):
            if idx == 5: break
            axes[idx].imshow(torch.transpose(image, 0, 2))
            axes[idx].set_title(f'label:{label}')
    plt.show()

if __name__ == "__main__":
    image_size = 224

    for i in range(1, 6):
        dataset = GTOSDataset(f'/fl_workspace/data/GTOS/labels/train{i}.txt',
                            transforms=transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(), 
                                ])
        )

        # data loader
        image_loader = DataLoader(dataset, 
                                batch_size  = 32, 
                                shuffle     = False, 
                                num_workers = 4,
                                pin_memory  = True)

        if i == 1:
            plot_samples(image_loader)

        compute_image_mean(image_loader)

# %%
