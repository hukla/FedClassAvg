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

class FMDTransform():
    def __init__(self, image_size, if_train=True):
        if if_train:
            self.data_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomResizedCrop(image_size, scale=(.8, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.6617,0.6048,0.5353), (0.2336,0.2235,0.2467)),
                ])
        else:
            self.data_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.6617,0.6048,0.5353), (0.2336,0.2235,0.2467)),
                ])
        
    def __call__(self, img):
        return self.data_transform(img)

class FMDDataset(torch.utils.data.Dataset):
    def __init__(self, label_path, if_train=True, transforms=None,image_dir=None):
        if not image_dir:
            self.image_dir = '/'.join(label_path.split('/')[:-2])+'/images'
        else:
            self.image_dir = image_dir

        self.file_list = list()
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = FMDTransform(224, if_train=if_train)

        with open(label_path) as f:
            line = f.readline()
            while True:
                if not line: break
                if 'jpg' in line:
                    self.file_list.append(line[:-1])
                line = f.readline()

        self.images = list()
        self.labels = list()

        # load data
        for img_path in self.file_list:
            label = img_path.split('/')[0]
            self.labels.append(label)
        
        self.classes = np.unique(self.labels)
        
        # class to num
        self.label2idx = {u:i for i, u in enumerate(self.classes)}
        self.idx2label = np.array(self.classes)
        
        self.targets = np.array([self.label2idx[label] for label in self.labels])

        self.configs={'lr': 0.03, 'batch_size': 64}

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(os.path.join(self.image_dir, img_path)).convert("RGB")
        img = self.transforms(img)

        return img, self.targets[index]

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

    for i in range(1, 11):
        dataset = FMDDataset(f'/fl_workspace/data/FMD/labels/train{i}.txt',
                            transforms=transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(), 
                                ])
        )

        # data loader
        image_loader = DataLoader(dataset, 
                                batch_size  = 32, 
                                shuffle     = False, 
                                num_workers = 1,
                                pin_memory  = True)
        
        if i == 1:
            plot_samples(image_loader)

        compute_image_mean(image_loader)

# %%
