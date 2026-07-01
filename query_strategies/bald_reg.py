import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import logging
import pdb
import os
import sys
import time
from .strategy_otf import StrategyOTF
from CloudL9_dataset import CloudL9
from CloudS2_dataset import CloudS2
logger = logging.getLogger(__name__)

class BALD_reg(StrategyOTF):
    def __init__(self, dataset, idxs_lb, net, handler, args, nb_inferences=10):
        super(BALD_reg, self).__init__(dataset, idxs_lb, net, handler, args)
        self.round_no = 0
        self.reg = net
        self.Y_train = dataset.get_target(use_threshold=False)
        self.nb_inferences = nb_inferences
        
        

    def query(self, N):
        
        batch_size = 256
        count = 0
        sampled_idxs = []
        
        inds = np.where(self.idxs_lb==0)[0]
        candidate_idxs = self.dataset.selected_ids[inds]
        
        
        
        if self.args["data"] in 'CloudL9':
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"])
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "CloudS2":
            candidate_feature_dataset = CloudS2(candidate_idxs, self.args["path"])
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "CloudL9_128": # NOTE: 13 January 2025
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"])
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "CloudS2_128": # NOTE: 13 January 2025
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"])
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "CloudL8_128": # NOTE: 13 January 2025
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"])
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "CloudSEN12_128": # NOTE: 26 June 2025
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"], img_shape=(3, 128, 128))
            Y_gt_candidate_dataset = candidate_feature_dataset.get_target(use_threshold=False) # NOTE: 21 Feb 2025
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "LandCoverAI_128": # NOTE: 26 June 2025
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"], img_shape=(3, 128, 128))
            Y_gt_candidate_dataset = candidate_feature_dataset.get_target(use_threshold=False) # NOTE: 21 Feb 2025
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        else:
            raise NotImplementedError
            
        self.reg.cuda()
        self.reg.eval()
        Y_train_candidate = torch.zeros(len(candidate_idxs), device='cuda')
        variance_candidate = torch.zeros(len(candidate_idxs), device='cuda')
        
        with torch.no_grad():
            for idxs, data  in enumerate(loader_te):               
                x, y, _ = data
                
                # Handle the problem of Cloud9 dataset
                x = x.squeeze(1)
                y = y.squeeze(-1)
                
                if self.args['net_type'] in ["gaussian_resnet18", "gaussian_net_semi_supervised"]:
                    x = Variable(x.cuda())
                    out = self.reg(x)
                    mu, sigma = torch.chunk(out, 2, dim=-1)
                    mu = mu.squeeze(-1)
                    sigma = sigma.squeeze(-1)
                    # preds = out.squeeze(-1)
                    if (idxs+1)*batch_size <= len(Y_train_candidate):
                        Y_train_candidate[idxs*batch_size:(idxs+1)*batch_size] = mu
                        variance_candidate[idxs*batch_size:(idxs+1)*batch_size] = sigma
                    else:
                        Y_train_candidate[idxs*batch_size:] = mu
                        variance_candidate[idxs*batch_size:] = sigma
                elif self.args['net_type'] in ["evidential_resnet18", "evidential_net_semi_supervised", "evidential_math_regressor", "evidential_math_regressor_SS"]:
                    x = Variable(x.cuda())
                    out = self.reg(x)
                    gamma, v, alpha, beta = torch.chunk(out, 4, dim=-1)
                    gamma = gamma.squeeze(-1)
                    v = v.squeeze(-1)
                    alpha = alpha.squeeze(-1)
                    beta = beta.squeeze(-1)
                    
                    aleatoric_uncertainty = beta / (alpha - 1)
                    epistemic_uncertainty = beta / (v*(alpha - 1))
                    
                    if (idxs+1)*batch_size <= len(Y_train_candidate):
                        Y_train_candidate[idxs*batch_size:(idxs+1)*batch_size] = gamma
                        variance_candidate[idxs*batch_size:(idxs+1)*batch_size] = aleatoric_uncertainty + epistemic_uncertainty
                    else:
                        Y_train_candidate[idxs*batch_size:] = gamma
                        variance_candidate[idxs*batch_size:] = aleatoric_uncertainty + epistemic_uncertainty
                elif self.args['net_type'] in ["beta_resnet18", "beta_net_semi_supervised", "beta_math_regressor", "beta_math_regressor_SS"]:
                    x = Variable(x.cuda())
                    out = self.reg(x)
                    mu, nu = torch.chunk(out, 2, dim=-1)
                    mu = mu.squeeze(-1)
                    nu = nu.squeeze(-1)
                    
                    entropy = torch.lgamma(mu*nu) + torch.lgamma((1-mu)*nu) - torch.lgamma(nu) + (nu-2)*torch.digamma(nu) - (mu*nu-1)*torch.digamma(mu*nu) - ((1-mu)*nu-1)*torch.digamma((1-mu)*nu)
                    
                    if (idxs+1)*batch_size <= len(Y_train_candidate):
                        Y_train_candidate[idxs*batch_size:(idxs+1)*batch_size] = mu
                        variance_candidate[idxs*batch_size:(idxs+1)*batch_size] = entropy
                    else:
                        Y_train_candidate[idxs*batch_size:] = mu
                        variance_candidate[idxs*batch_size:] = entropy
                elif self.args['net_type'] in ["bayesian_resnet18", "bayesian_net_semi_supervised", "mcdropout_math_regressor", "mcdropout_math_regressor_SS", "bayesian_mobilenetv3", "bayesian_mobilenetv4"]: # MONTE-CARLO DROPOUT           
                    x = Variable(x.cuda())
                    out = self.reg(x, k=self.nb_inferences)
                    preds = out.squeeze(-1)
                    if (idxs+1)*batch_size <= len(Y_train_candidate):
                        Y_train_candidate[idxs*batch_size:(idxs+1)*batch_size] = preds.mean(dim=1)
                        variance_candidate[idxs*batch_size:(idxs+1)*batch_size] = torch.std(preds, dim=1)
                    else:
                        Y_train_candidate[idxs*batch_size:] = preds.mean(dim=1)
                        variance_candidate[idxs*batch_size:] = torch.std(preds, dim=1)
                else:
                    raise NotImplementedError
                        
                Y_train_candidate = Y_train_candidate.detach().cpu()
    
                # Sort the variances in the descending order
                variance_candidate = variance_candidate.cpu().numpy()
                sorted_idxs = np.argsort(variance_candidate)[::-1]
                
                selected_samples = inds[sorted_idxs][:N]
                print("selected_samples: ")
                print(selected_samples)
                print("selected images in the directory:")
                print(self.dataset.selected_ids[selected_samples])
            
                return selected_samples