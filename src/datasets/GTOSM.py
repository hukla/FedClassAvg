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

class GTOSMTransform():
    def __init__(self, image_size):
        self.data_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4592, 0.3945, 0.3436], 
                std=[0.2217, 0.2081, 0.1995]), 
            ])
    
    def __call__(self, img):
        return self.data_transform(img)

class GTOSMDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, transforms=None):
        self.file_list = list()
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = GTOSMTransform(224)

        self.images = list()
        self.labels = list()

        # make dataset
        self.classes = os.listdir(image_path)
        for cls in self.classes:
            cls_path = os.path.join(image_path, cls)
            for img in os.listdir(cls_path):
                self.file_list.append(os.path.join(cls_path, img))
                self.labels.append(cls)
            
        # self.classes = np.unique(self.labels)
        
        # class to num
        self.label2idx = {u:i for i, u in enumerate(self.classes)}
        self.idx2label = np.array(self.classes)
        
        self.targets = np.array([self.label2idx[label] for label in self.labels])
        self.configs={'lr': 0.0003, 'batch_size': 64}

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # load data
        img_path = self.file_list[index]
        label = self.targets[index]
    
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        # if img.shape != torch.Size([3, 224, 224]):
            # img = torch.cat([img, img, img], axis=0)           

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

    dataset = GTOSMDataset('/fl_workspace/data/gtos-mobile/train',
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

    # if i == 1:
        # plot_samples(image_loader)

    compute_image_mean(image_loader)

# %%
