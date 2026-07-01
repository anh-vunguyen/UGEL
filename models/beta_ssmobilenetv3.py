import torch
import torch.nn as nn
import torch.nn.functional as F
from beta_mobilenetv3 import beta_mobilenetv3_small

class Beta_MobileNetv3_SS(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.branch1 = beta_mobilenetv3_small()
        self.branch2 = beta_mobilenetv3_small()
        
    def forward(self, data, step=1):
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)
