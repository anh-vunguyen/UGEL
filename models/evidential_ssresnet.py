import torch
import torch.nn as nn
import torch.nn.functional as F
from models.evidential_resnet import evidential_resnet18

class Evidential_Network_SS(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.branch1 = evidential_resnet18(num_classes=num_classes)
        self.branch2 = evidential_resnet18(num_classes=num_classes)
        
    def forward(self, data, step=1):
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)
