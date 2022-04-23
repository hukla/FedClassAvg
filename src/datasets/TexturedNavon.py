#%%
from re import S
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

class TexturedNavonTransform():
    def __init__(self, image_size, if_train=True):
        if if_train:
            self.data_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomResizedCrop(image_size, scale=(.8, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5288,0.4731,0.4247), (0.2625,0.2531,0.2607)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #https://github.com/jiaxue1993/pytorch-material-classification/blob/master/dataloader/dtd.py
                ])
        else:
            self.data_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #https://github.com/jiaxue1993/pytorch-material-classification/blob/master/dataloader/dtd.py
                ])
    
    def __call__(self, img):
        return self.data_transform(img)

class TexturedNavonDataset(torch.utils.data.Dataset):
    def __init__(self, label_path, target_label_idx, transforms=None, image_dir=None, image_size=224):
        self.target_label_idx = target_label_idx
        if not image_dir:
            self.image_dir = '/'.join(label_path.split('/')[:-2])+'/images'
        else:
            self.image_dir = image_dir

        self.file_list = list()
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = TexturedNavonTransform(image_size)

        with open(label_path) as f:
            line = f.readline()
            while True:
                if not line: break
                self.file_list.append(line[:-1])
                line = f.readline()

        self.labels = []

        # load data
        for img_path in self.file_list:
            shape = img_path.split('_')[0]
            texture = img_path.split('_')[1]
            
            if self.target_label_idx == 0: # shape
                self.labels.append(shape)
            else:
                self.labels.append(texture)
        
        self.classes = np.unique(self.labels)
        
        # class to num
        self.label2idx = {u:i for i, u in enumerate(self.classes)}
        self.idx2label = np.array(self.classes)
        
        self.targets = np.array([self.label2idx[label] for label in self.labels])

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(os.path.join(self.image_dir, img_path))
        img = self.transforms(img)

        return img, self.targets[index]

def compute_image_mean(label_path, target_label_idx):
    image_size = 224
    dataset = TexturedNavonDataset(label_path, target_label_idx,
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

    fig, axes = plt.subplots(nrows=1, ncols=5)
    for idx, (image, label) in enumerate(dataset):
        if idx == 5: break
        axes[idx].imshow(torch.transpose(image, 0, 2))
        axes[idx].set_title(f'label:{label}')

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

if __name__ == "__main__":
    image_size = 128
    dataset = TexturedNavonDataset('data/navon-dtd/imsize224_shape200/labels/train1.txt', 0, image_size=image_size)

    # data loader
    image_loader = DataLoader(dataset, 
                            batch_size  = 32, 
                            shuffle     = False, 
                            num_workers = 4,
                            pin_memory  = True)

    # fig, axes = plt.subplots(nrows=1, ncols=5)
    # for images, labels in image_loader:
        # print(images, labels)
        # for idx, (image, label) in enumerate(zip(images, labels)):
            # if idx == 5: break
            # axes[idx].imshow(torch.transpose(image, 0, 2))
            # axes[idx].set_title(f'label:{label}')
    for i in range(1, 11):
        compute_image_mean(f'data/navon-dtd/imsize224_shape200/labels/train{i}.txt', 0)

# %%
