import torch
from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor

import models.mc_dropout as mc_dropout

class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes=1):
        super().__init__(num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5,)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc1 = nn.Linear(512, 128)
        self.fc1_drop = mc_dropout.MCDropout2d()
        self.fc2 = nn.Linear(128, num_classes)
        
        
    def mc_forward_impl(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        
        
        x = x.view(-1, 512)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        
        return x