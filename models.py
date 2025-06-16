import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeconvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # Deconv layers
        self.deconv1 = nn.ConvTranspose2d(16, 6, 5)
        self.deconv2 = nn.ConvTranspose2d(6, 3, 5)
        self.unpool = nn.MaxUnpool2d(2, 2)

        # Fully-Connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv part
        x = F.relu(self.conv1(x))
        size1 = x.size()
        x, indices1 = self.pool(x)

        x = F.relu(self.conv2(x))
        size2 = x.size()
        x, indices2 = self.pool(x)

        # Fully-Connected part
        y = torch.flatten(x, 1) # flatten all dimensions except batch
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        # Deconv part
        x_ = self.deconv1(F.relu(self.unpool(x, indices2, output_size=size2)))
        x_ = self.deconv2(F.relu(self.unpool(x_, indices1, output_size=size1)))

        return y, x_