import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenet.beta_mobilenetv4 import BetaMobileNetV4

class Beta_MobileNetv4_SS(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.branch1 = BetaMobileNetV4("MobileNetV4ConvSmall")
        self.branch2 = BetaMobileNetV4("MobileNetV4ConvSmall")
        
    def forward(self, data, step=1):
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)
