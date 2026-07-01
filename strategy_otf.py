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
import time
from CloudL9_dataset import CloudL9
from CloudS2_dataset import CloudS2
from math_functions import Sigmoid_Sum, Oscillatory
from uci_regression_datasets import UCI_Dataset
logger = logging.getLogger(__name__)

# NOTE: 12 04 2025
from losses import Gaussian_NLL, EvidentialRegressionLoss, Beta_NLL, beta_nll_loss, EvidentialRegressionLoss_v2

# NOTE: 06 02 2025
import torchvision

# Regression
class StrategyOTF:
    def __init__(self, dataset, idxs_lb, net, handler, args):
        self.dataset = dataset # Load each frame/image on-the-fly
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args 
        self.n_pool = len(dataset)
        self.round_no = 0

        # Training
        self.training_info = {} # Saving intermediate training info
        # Testing info
        self.test_info = {} # Saving intermediate test info
        use_cuda = torch.cuda.is_available()
        self.reg = self.net.cuda() # Regressor

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.reg.train()
        mseFinal = 0.
        
        # losses
        running_loss = 0
    
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            if len(y.shape) == 2:
                y = y.squeeze(-1)
            
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            
            if self.args['net_type'] in ['pretrainedS2_resnet50' , 'pretrainedL9_resnet50', "vanilla_resnet50"]:
                out = self.reg(x)
            elif self.args['net_type'] in ["bayesian_net", "bayesian_resnet18", "bayesian_mobilenetv3", "bayesian_mobilenetv4"]:
                out = self.reg(x, k=1)
                out = out.squeeze(-1)
            elif self.args['net_type'] in ["gaussian_resnet18", "gaussian_net_semi_supervised"]: 
                mu, sigma = torch.chunk(out, 2, dim=-1)
                mu = mu.squeeze(-1)
                sigma = sigma.squeeze(-1)
            elif self.args['net_type'] in ["evidential_resnet18", "evidential_net_semi_supervised"]:
                out = self.reg(x)
                mu, v, alpha, beta = torch.chunk(out, 4, dim=-1)
                mu = mu.squeeze(-1)
                v = v.squeeze(-1)
                alpha = alpha.squeeze(-1)
                beta = beta.squeeze(-1)
            elif self.args['net_type'] in ['beta_resnet18', "beta_net_semi_supervised", "beta_math_regressor", "beta_math_regressor_SS", "beta_mobilenetv3", "beta_mobilenetv3_semi_supervised", "beta_mobilenetv4", "beta_mobilenetv4_semi_supervised"]:
                out = self.reg(x)
                mu, nu = torch.chunk(out, 2, dim=-1)
                mu = mu.squeeze(-1)
                nu = nu.squeeze(-1)
            elif self.args['net_type'] in ["beta_net_semi_supervised_2",]:
                out = self.reg(x)
                y_hat, mu, nu = torch.chunk(out, 3, dim=-1)
                y_hat = y_hat.squeeze(-1)
                mu = mu.squeeze(-1)
                nu = nu.squeeze(-1)
            elif self.args['net_type'] in ["resnet18_semi_supervised"]: 
                out = self.reg(x)
            else :
                out = self.reg(x)
    
            out = out.squeeze(-1)
                        
            if self.args['net_type'] in ["gaussian_resnet18", "gaussian_net_semi_supervised"]:
                loss_nll = beta_nll_loss(mu, sigma, y, beta=0.75)
                loss_rmse = torch.sqrt(F.mse_loss(mu, y))
                loss = loss_rmse + loss_nll
            elif self.args['net_type'] in ["evidential_resnet18", "evidential_net_semi_supervised"]:
                loss = EvidentialRegressionLoss(y, out)
                mu, v, alpha, beta = torch.chunk(out, 4, dim=-1)
                mu = mu.squeeze(-1)
                v = v.squeeze(-1)
                alpha = alpha.squeeze(-1)
                beta = beta.squeeze(-1)
                epistemic_loss = beta/(v*(alpha-1)+1e-6)
                N = self.idxs_lb.sum()
                loss = loss + (epistemic_loss.mean()/N)
            elif self.args['net_type'] in ["beta_resnet18", "beta_net_semi_supervised", "beta_mobilenetv3", "beta_mobilenetv3_semi_supervised", "beta_mobilenetv4_semi_supervised"]:
                loss_nll = Beta_NLL(y, mu, nu)
                loss_rmse = torch.sqrt(F.mse_loss(mu, y))
                loss = loss_rmse + loss_nll
            elif self.args['net_type'] in ['beta_net_semi_supervised_2']:
                loss_nll = Beta_NLL(y, mu, nu)
                loss_rmse = torch.sqrt(F.mse_loss(y_hat, y))
                loss = loss_rmse + loss_nll    
            else:     
                loss = torch.sqrt(F.mse_loss(out, y))    
            
            # Training losses
            running_loss += loss.detach()
            
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.reg.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
            optimizer.step()
            
        self.training_info[f"epoch_{epoch}"] = {"loss": running_loss.item() / len(loader_tr)}
            
        return running_loss / len(loader_tr)
    
    
    def train(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        print(">"*10 + "TRAINING" + "<"*10) 
        logger.info(">"*10 + "TRAINING" + "<"*10)
        
        n_epoch = self.args['n_epoch']
        if self.args['net_type'] == 'pretrained': 
            self.reg = self.args['pretrained_resnet']().cuda()
        elif self.args['net_type'] == 'pretrained_resnet18':
            self.reg = self.args['pretrained_resnet18']().cuda()
        elif self.args['net_type'] == 'bayesian_resnet18':
            self.reg = self.args['bayesian_resnet18']().cuda()
        elif self.args['net_type'] == 'bayesian_net_semi_supervised':
            self.reg = self.args['bayesian_net_semi_supervised']()
            self.reg.branch1.cuda()
            self.reg.branch2.cuda()
        elif self.args['net_type'] == 'bayesian_resnet18_cloudscout':
            self.reg = self.args['bayesian_resnet18_cloudscout']()
            self.reg.branch1.cuda()
            self.reg.branch2.cuda()
        elif self.args['net_type'] == 'gaussian_resnet18':
            self.reg = self.args['gaussian_resnet18']().cuda()
        elif self.args['net_type'] == 'gaussian_net_semi_supervised':
            self.reg = self.args['gaussian_net_semi_supervised']()
            self.reg.branch1.cuda()
            self.reg.branch2.cuda()
        elif self.args['net_type'] == 'evidential_resnet18':
            self.reg = self.args['evidential_resnet18']().cuda()
        elif self.args['net_type'] == 'evidential_net_semi_supervised':
            self.reg = self.args['evidential_net_semi_supervised']().cuda()
        elif self.args['net_type'] == 'beta_resnet18':
            self.reg = self.args['beta_resnet18']().cuda()
        elif self.args['net_type'] == 'beta_net_semi_supervised_2':
            self.reg = self.args['beta_net_semi_supervised_2']().cuda()
        elif self.args['net_type'] == "resnet18_semi_supervised":
            self.reg = self.args['resnet18_semi_supervised']().cuda()
        elif self.args['net_type'] == "bayesian_mobilenetv3":
            self.reg = self.args['bayesian_mobilenetv3']().cuda()
        elif self.args['net_type'] == "mobilenetv3":
            self.reg = self.args['mobilenetv3']().cuda()
        elif self.args['net_type'] == "beta_mobilenetv3":
            self.reg = self.args['beta_mobilenetv3']().cuda()
        elif self.args['net_type'] == "beta_mobilenetv3_semi_supervised":
            self.reg = self.args['beta_mobilenetv3_semi_supervised']().cuda()
        elif self.args['net_type'] == "mobilenetv4":
            self.reg = self.args['mobilenetv4']().cuda()
        elif self.args['net_type'] == "bayesian_mobilenetv4":
            self.reg = self.args['bayesian_mobilenetv4']().cuda()
        elif self.args['net_type'] == "beta_mobilenetv4":
            self.reg = self.args['beta_mobilenetv4']().cuda()
        elif self.args['net_type'] == "beta_mobilenetv4_semi_supervised": 
            self.reg = self.args['beta_mobilenetv4_semi_supervised']().cuda()             
        else: 
            self.reg =  self.net.apply(weight_reset).cuda() # Regressor
            
        if self.args['net_type'] in ['resnet18_semi_supervised', 'bayesian_net_semi_supervised', 'bayesian_resnet18_cloudscout', "gaussian_net_semi_supervised", "evidential_net_semi_supervised", "beta_net_semi_supervised", "mcdropout_math_regressor_SS", "evidential_math_regressor_SS", "beta_math_regressor_SS", "beta_mobilenetv3_semi_supervised", "beta_mobilenetv4_semi_supervised"]:
            start_time = time.perf_counter()
            self.train_with_supervised_learning(n_epoch)
            end_time = time.perf_counter()
            runtime = end_time - start_time
            print(f"Training executed in {runtime:.6f} seconds.")
        else:
            if self.args['use_same_init_weights']:
                print("Load same initialised weights")
                init_checkpoint = torch.load(os.path.join(self.args['full_logging_dir_name'], 'init_weights.pt'))
                self.reg.load_state_dict(init_checkpoint['model_state_dict'])
                
            
            optimizer = optim.Adam(self.reg.parameters(), lr = self.args['lr'])      
            
            if 'Cloud' in self.args['data']:
                idxs_train = self.dataset.selected_ids[self.idxs_lb]
            else:
                idxs_train = np.arange(self.n_pool)[self.idxs_lb]
            
            self.training_info["idxs_train"] = idxs_train

            if self.args['data'] == "CloudL9":
                self.dataset = CloudL9(idxs_train, self.args["path"])
                loader_tr = DataLoader(self.handler(self.dataset), shuffle=True, **self.args['loader_tr_args'])
            elif self.args['data'] == "CloudS2":
                self.dataset = CloudS2(idxs_train, self.args["path"])
                loader_tr = DataLoader(self.handler(self.dataset), shuffle=True, **self.args['loader_tr_args'])     
            elif self.args['data'] == "CloudL9_128":
                dataset = CloudL9(idxs_train, self.args["path"], img_shape=(3, 128, 128))
                loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            elif self.args['data'] == "CloudS2_128":
                dataset = CloudS2(idxs_train, self.args["path"], img_shape=(3, 128, 128))
                loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            elif self.args['data'] == "CloudL8_128":
                dataset = CloudL9(idxs_train, self.args["path"], img_shape=(3, 128, 128))
                loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            elif self.args['data'] == "CloudSEN12_128":
                dataset = CloudL9(idxs_train, self.args["path"], img_shape=(3, 128, 128))
                loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            elif self.args['data'] == "LandCoverAI_128": 
                dataset = CloudL9(idxs_train, self.args["path"], img_shape=(3, 128, 128))
                loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            else:
                loader_tr = DataLoader(self.handler(self.dataset[idxs_train]), shuffle=True, **self.args['loader_tr_args'])
    
            epoch = 0
            lossCurrent = 0.
            
            start_time = time.perf_counter()
            while epoch < n_epoch: 
                lossCurrent = self._train(epoch, loader_tr, optimizer)
                if epoch % 1 == 0:
                    print(str(epoch) + ' Training error: ' + str(lossCurrent.item()), flush=True)
                    logger.info(str(epoch) + ' Training error: ' + str(lossCurrent.item()))
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"Epoch {epoch} - Learning rate: {lr}")
                epoch += 1
            end_time = time.perf_counter()
            runtime = end_time - start_time
            print(f"Training executed in {runtime:.6f} seconds.")
    
    
    def predict(self, test_dataset, for_testing=True):
        if for_testing:
            print(">"*10 + "TESTING" + "<"*10)
            logger.info(">"*10 + "TESTING" + "<"*10)
        else:
            print(">"*10 + "VALIDATING" + "<"*10)
            logger.info(">"*10 + "VALIDATING" + "<"*10)
        idxs_te = np.arange(len(test_dataset))
        loader_te = DataLoader(self.handler(test_dataset), shuffle=False, **self.args['loader_te_args'])
        
        if  self.args['net_type'] in ["bayesian_net", "bayesian_net_semi_supervised", "bayesian_resnet18_cloudscout", "gaussian_net_semi_supervised", "mcdropout_math_regressor_SS"]:
            self.reg.branch1.eval()
            self.reg.branch2.eval()
        else:
            self.reg.eval()
        outs = torch.zeros(len(idxs_te))
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                if self.args['net_type'] in ['pretrainedS2_resnet50' , 'pretrainedL9_resnet50']:
                    out = self.reg(x)
                    outs[idxs] = out.squeeze(-1).cpu()
                elif self.args['net_type'] in ["bayesian_net", "bayesian_resnet18", "bayesian_net_semi_supervised", "bayesian_resnet18_cloudscout", "mcdropout_math_regressor", "mcdropout_math_regressor_SS", "bayesian_mobilenetv3", "bayesian_mobilenetv4"]:
                    out = self.reg(x, k=10)
                    out = out.mean(dim=1)
                    outs[idxs] = out.squeeze(-1).cpu()
                elif self.args['net_type'] in ["gaussian_resnet18", "gaussian_net_semi_supervised"]:
                    out = self.reg(x)
                    mu, sigma = torch.chunk(out, 2, dim=-1)
                    mu = mu.squeeze(-1)
                    sigma = sigma.squeeze(-1)
                    outs[idxs] = mu.cpu()
                elif self.args['net_type'] in ['evidential_resnet18', 'evidential_net_semi_supervised', 'evidential_math_regressor', 'evidential_math_regressor_SS']:
                    out = self.reg(x)
                    mu, v, alpha, beta = torch.chunk(out, 4, dim=-1)
                    mu = mu.squeeze(-1)
                    v = v.squeeze(-1)
                    alpha = alpha.squeeze(-1)
                    beta = beta.squeeze(-1)
                    outs[idxs] = mu.cpu()
                elif self.args['net_type'] in ['beta_resnet18', "beta_net_semi_supervised", "beta_math_regressor", "beta_math_regressor_SS", "beta_mobilenetv3", "beta_mobilenetv3_semi_supervised", "beta_mobilenetv4_semi_supervised"]:
                    out = self.reg(x)
                    mu, nu = torch.chunk(out, 2, dim=-1)
                    mu = mu.squeeze(-1)
                    nu = nu.squeeze(-1)
                    outs[idxs] = mu.cpu()
                elif self.args['net_type'] in ["beta_net_semi_supervised_2"]:
                    out = self.reg(x)
                    y_hat, mu, nu = torch.chunk(out, 3, dim=-1)
                    y_hat = y_hat.squeeze(-1)
                    mu = mu.squeeze(-1)
                    nu = nu.squeeze(-1)
                    outs[idxs] = y_hat.cpu()
                elif self.args['net_type'] in ["resnet18_semi_supervised"]:
                    out = self.reg(x)
                    outs[idxs] = out.squeeze(-1).cpu()
                else:
                    out = self.reg(x)
                    outs[idxs] = out.squeeze(-1).cpu()
        return outs
