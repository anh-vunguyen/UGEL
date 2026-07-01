import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_mobilenetv3 import bayesian_mobilenetv3_small

class bayesian_mobilenetv3_(nn.Module):
     def __init__(self, num_classes=1):
        super(bayesian_mobilenetv3_, self).__init__()
        self.net = bayesian_mobilenetv3_small()
        self.k = 10
        
     def forward(self, x, k=1):
        if k == 1:
            out = torch.sigmoid(self.net(x))
            return out
        else:
            self.net.enable_mc_dropout()
            outs = []
            for i in range(k):
                out = torch.sigmoid(self.net(x))
                outs.append(out.squeeze(-1))
            outs = torch.vstack(outs)
            outs = outs.permute(1, 0)
            return outs

class Bayesian_Network_SS(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.branch1 = bayesian_mobilenetv3_()
        self.branch2 = bayesian_mobilenetv3_()
        
    def forward(self, data, k=1, step=1, return_feature=False):
        if not self.training:
            pred1 = self.branch1(data, k=k)
            return pred1
        else:
            if step == 1:
                return self.branch1(data, k=k)
            elif step == 2:
                return self.branch2(data, k=k)