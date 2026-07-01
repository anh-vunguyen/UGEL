import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_resnet import bayesian_resnet18
from cloudscout import CloudScout

class bayesian_resnet18_(nn.Module):
     def __init__(self, num_classes=1):
        super(bayesian_resnet18_, self).__init__()
        self.net = bayesian_resnet18(num_classes=1)
        self.k = 10
        
     def forward(self, x, k=1):
        if k == 1:
            out = torch.sigmoid(self.net(x))
            return out
        else:
            outs = []
            for i in range(k):
                out = torch.sigmoid(self.net(x))
                outs.append(out.squeeze(-1))
            outs = torch.vstack(outs)
            outs = outs.permute(1, 0)
            return outs

class Bayesian_ResNet_CloudScout(nn.Module):
    def __init__(self, num_classes=1, cloudscout_checkpoint=None):
        super().__init__()
        self.branch1 = bayesian_resnet18_(num_classes)
        self.branch2 = CloudScout()
        if cloudscout_checkpoint:
            checkpoint = torch.load(cloudscout_checkpoint)
            self.branch2.load_state_dict(checkpoint)
            
        
    def forward(self, data, k=1, step=1):
        if not self.training:
            pred1 = self.branch1(data, k=k)
            
        if step == 1:
            return self.branch1(data, k=k)
        elif step == 2:
            return self.branch2(data)
        