#%%
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import string

def make_file_list(root_path):
    img_path_list = list()

    # for shape in label_list:
    for file in os.listdir(os.path.join(root_path)):
        if 'png' in file:
            img_path = os.path.join(root_path, file)
            img_path_list.append(img_path)

    return img_path_list

class NavonTransform():

    def __init__(self):
        self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
    
    def __call__(self, img):
        return self.data_transform(img)

class NavonDataset(torch.utils.data.Dataset):
    
    def __init__(self, file_list, transforms, target_class_idx):
        assert target_class_idx in [1, 2, 3], f'ERROR: undefined class idx {target_class_idx}'

        self.file_list = file_list
        if transforms:
            self.transform = transforms
        else:
            self.transform = NavonTransform()
        # self.transform = transform
        self.color2idx = {'blue': 1, 'red': 2, 'yellow': 3, 'green': 4, 'black': 0}
        self.int2color = {0: 'black', 1: 'blue', 2: 'red', 3: 'yellow', 4: 'green'}
        self.target_shapes = list()
        self.target_textures = list()
        self.target_colors = list()

        for img_path in file_list:
            img_name = img_path.split('/')[-1].split('.')[0].split('-')[0]
            shape = ord(img_name.split('_')[0]) - 65
            texture = ord(img_name.split('_')[1]) - 65
            color = self.color2idx[img_name.split('_')[2]]

            self.target_shapes.append(shape)
            self.target_textures.append(texture)
            self.target_colors.append(color)
        
        self.target_shapes = torch.tensor(self.target_shapes)
        self.target_textures = torch.tensor(self.target_textures)
        self.target_colors = torch.tensor(self.target_colors)

        if target_class_idx == 0:
            self.targets = self.target_shapes
        elif target_class_idx == 1:
            self.targets = self.target_textures
        elif target_class_idx == 2:
            self.targets = self.target_colors
        else:
            print('ERROR: undefined class idx ', target_class_idx)
            exit()

        if target_class_idx < 3:
            self.classes = list(string.ascii_uppercase)
        else:
            self.classes = list(self.color2idx.keys())

    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        return img_transformed, self.targets[index] 

if __name__ == "__main__":
    os.chdir('/fl_workspace/federated_pytorch')
    dataset = NavonDataset(make_file_list('data/navon/imsize128/shape110_texture8/train'), NavonTransform())
    print(len(dataset))
    fig, axes = plt.subplots(nrows=1, ncols=5)
    for idx, (image, label) in enumerate(dataset):
        if idx == 5: break
        axes[idx].imshow(torch.transpose(image, 0, 2))
        axes[idx].set_title(f'label:{label}')

# %%
