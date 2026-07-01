import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_resnet import bayesian_resnet18

class bayesian_resnet18_(nn.Module):
     def __init__(self, num_classes=1):
        super(bayesian_resnet18_, self).__init__()
        self.net = bayesian_resnet18(num_classes=1)
        self.k = 10
        
     def forward(self, x, k=1, return_feature=False):
        if return_feature:
            if k == 1:
                out, feat = self.net(x, return_feature=True)
                out = torch.sigmoid(out)
                return out, feat
            else:
                outs = []
                feats = []
                for i in range(k):
                    out, feat = self.net(x)
                    out = torch.sigmoid(out)
                    outs.append(out.squeeze(-1))
                    feats.append(feat.squeeze(-1))
                outs = torch.vstack(outs)
                feats = torch.vstack(feats)
                outs = outs.permute(1, 0)
                feats = feats.permute(1, 0)
                return outs, feats
        else: 
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

class Bayesian_Network_SS(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.branch1 = bayesian_resnet18_(num_classes)
        self.branch2 = bayesian_resnet18_(num_classes)
        
    def forward(self, data, k=1, step=1, return_feature=False):
        if not self.training:
            pred1 = self.branch1(data, k=k)
            return pred1
        if not return_feature:
            if step == 1:
                return self.branch1(data, k=k)
            elif step == 2:
                return self.branch2(data, k=k)
        else: # NOTE: 17/02/2025
            if step == 1:
                return self.branch1(data, k=k, return_feature=True)
            elif step == 2:
                return self.branch2(data, k=k, return_feature=True)
        