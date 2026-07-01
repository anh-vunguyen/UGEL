import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18 import resnet18

class ResNet18_SS(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.branch1 = resnet18(num_classes=num_classes)
        self.branch2 = resnet18(num_classes=num_classes)
        
    def forward(self, data, step=1):
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)
