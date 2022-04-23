import torch
import torch.nn as nn
from collections import OrderedDict

class alexnet_mnist(nn.Module):  
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding='same'),
            nn.ReLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3,3))
            

        self.fc = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout(p=0.5)),
            ('fcin', nn.Linear(256 * 3 * 3, 512)),
            ('relu', nn.ReLU()),
            # ('drop2', nn.Dropout(p=0.5)),
            ('fcout', nn.Linear(512, num_classes)),
        ]))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc(out)
        return out