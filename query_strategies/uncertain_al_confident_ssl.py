import numpy as np
from copy import deepcopy
import logging
import pdb
import os
import sys
import time
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from .strategy_otf import StrategyOTF
from CloudL9_dataset import CloudL9
from CloudS2_dataset import CloudS2
from losses import Gaussian_NLL, EvidentialRegressionLoss, Beta_NLL
logger = logging.getLogger(__name__)

# Uncertainty-Guided Edge Learning
class Uncertain_AL_Confident_SSL(StrategyOTF):
    def __init__(self, dataset, idxs_lb, net, handler, args, nb_inferences=10, window_size=100):
        super(Uncertain_AL_Confident_SSL, self).__init__(dataset, idxs_lb, net, handler, args)
        self.round_no = 0
        self.reg = net
        self.Y_train = dataset.get_target(use_threshold=False)
        self.nb_inferences = 10
        self.next_round_ssl_idxs = None
        
        
    def query(self, N):
        batch_size = 256 # Please feel free to modify this parameter to match your available GPU memory.
        
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
        elif self.args["data"] in "CloudL9_128":
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"])
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "CloudS2_128":
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"])
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "CloudL8_128":
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"], img_shape=(3, 128, 128))
            Y_gt_candidate_dataset = candidate_feature_dataset.get_target(use_threshold=False) # NOTE: 21 Feb 2025
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "CloudSEN12_128":
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"], img_shape=(3, 128, 128))
            Y_gt_candidate_dataset = candidate_feature_dataset.get_target(use_threshold=False) # NOTE: 21 Feb 2025
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        elif self.args["data"] in "LandCoverAI_128":
            candidate_feature_dataset = CloudL9(candidate_idxs, self.args["path"], img_shape=(3, 128, 128))
            Y_gt_candidate_dataset = candidate_feature_dataset.get_target(use_threshold=False) # NOTE: 21 Feb 2025
            loader_te = DataLoader(self.handler(candidate_feature_dataset, transform=self.args['transformTest']),
                                batch_size=batch_size,
                                shuffle=False,)
        else:
            raise NotImplementedError
        

        start_time = time.perf_counter()
        self.reg.cuda()
        self.reg.eval()
        Y_train_candidate = torch.zeros(len(candidate_idxs), device='cuda')
        variance_candidate = torch.zeros(len(candidate_idxs), device='cuda')
        
        with torch.no_grad():
            for idxs, data  in enumerate(loader_te):               
                x, y, _ = data
                
                if self.args['data'] == "CloudL9":
                    x = x.squeeze(1)
                    y = y.squeeze(-1)
                
                x = Variable(x.cuda())
                
                if self.args['net_type'] in ["bayesian_net", "bayesian_resnet18", "bayesian_net_semi_supervised", "bayesian_resnet18_cloudscout", "mcdropout_math_regressor", "mcdropout_math_regressor_SS"]:
                    out = self.reg(x, k=self.nb_inferences)
                    preds = out.squeeze(-1)
                    if (idxs+1)*batch_size <= len(Y_train_candidate):
                        Y_train_candidate[idxs*batch_size:(idxs+1)*batch_size] = preds.mean(dim=1)
                        variance_candidate[idxs*batch_size:(idxs+1)*batch_size] = torch.std(preds, dim=1)
                    else:
                        Y_train_candidate[idxs*batch_size:] = preds.mean(dim=1)
                        variance_candidate[idxs*batch_size:] = torch.std(preds, dim=1)
                elif self.args['net_type'] in ['gaussian_resnet18', "gaussian_net_semi_supervised"]:
                    out = self.reg(x)
                    mu, sigma = torch.chunk(out, 2, dim=-1)
                    mu = mu.squeeze(-1)
                    sigma = sigma.squeeze(-1)
                    if (idxs+1)*batch_size <= len(Y_train_candidate):
                        Y_train_candidate[idxs*batch_size:(idxs+1)*batch_size] = mu
                        variance_candidate[idxs*batch_size:(idxs+1)*batch_size] = sigma
                    else:
                        Y_train_candidate[idxs*batch_size:] = mu
                        variance_candidate[idxs*batch_size:] = sigma
                elif self.args['net_type'] in ["evidential_resnet18", "evidential_net_semi_supervised", "evidential_math_regressor_SS"]:
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
                elif self.args['net_type'] in ["beta_resnet18", "beta_net_semi_supervised", "beta_math_regressor", "beta_math_regressor_SS", "beta_mobilenetv3_semi_supervised", "beta_mobilenetv4_semi_supervised"]:
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
                elif self.args['net_type'] in ["beta_net_semi_supervised_2"]:
                    
                    x = Variable(x.cuda())
                    out = self.reg(x)
                    pred, mu, nu = torch.chunk(out, 3, dim=-1)
                    pred = pred.squeeze(-1)
                    mu = mu.squeeze(-1)
                    nu = nu.squeeze(-1)
                    
                    entropy = torch.lgamma(mu*nu) + torch.lgamma((1-mu)*nu) - torch.lgamma(nu) + (nu-2)*torch.digamma(nu) - (mu*nu-1)*torch.digamma(mu*nu) - ((1-mu)*nu-1)*torch.digamma((1-mu)*nu)
                    
                    if (idxs+1)*batch_size <= len(Y_train_candidate):
                        Y_train_candidate[idxs*batch_size:(idxs+1)*batch_size] = mu
                        variance_candidate[idxs*batch_size:(idxs+1)*batch_size] = entropy
                    else:
                        Y_train_candidate[idxs*batch_size:] = mu
                        variance_candidate[idxs*batch_size:] = entropy                  
                else:
                    raise NotImplementedError
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Uncertainty estimation executed in {runtime:.6f} seconds.")
        Y_train_candidate = Y_train_candidate.detach().cpu()
        
        variance_candidate = variance_candidate.cpu().numpy()
        sorted_idxs = np.argsort(variance_candidate)[::-1]
        
        np.save(os.path.join(self.args["full_logging_dir_name"], f"variance_rd{self.round_no}.npy"), variance_candidate)
        self.round_no += 1
        
        selected_samples = inds[sorted_idxs][:N]

        print("Using most confident predictions for Semi-Supvervised Learning")
        nb_of_samples_for_semi = self.idxs_lb.sum() + N
        self.next_round_ssl_idxs = inds[sorted_idxs][-nb_of_samples_for_semi:]
        
        return selected_samples

    
    def train_with_supervised_learning(self, n_epoch):
        optimizer_l = optim.Adam(self.reg.branch1.parameters(), lr = self.args['lr'])    
        optimizer_r = optim.Adam(self.reg.branch2.parameters(), lr = self.args['lr'])

        idxs_train = self.dataset.selected_ids[self.idxs_lb] # NOTE: 24 January 2025
        idxs_train_semisupervised = self.next_round_ssl_idxs
        
        self.training_info["idxs_train"] = idxs_train
        self.training_info["idxs_train_semisupervised"] = idxs_train_semisupervised
        
        if self.args['data'] == "CloudL9_128":
            dataset = CloudL9(idxs_train, self.args["path"], img_shape=(3, 128, 128))
            loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            if idxs_train_semisupervised is not None:
                dataset_semisupervised = CloudL9(idxs_train_semisupervised, self.args["path"], img_shape=(3, 128, 128))
                loader_tr_semisupervised = DataLoader(self.handler(dataset_semisupervised), shuffle=True, **self.args['loader_tr_args'])
            else:
                epoch = 0
                lossCurrent = 0.
                while epoch < n_epoch: 
                    lossCurrent = self._train(epoch, loader_tr, optimizer_l)
                    if epoch % 1 == 0:
                        print(str(epoch) + ' Training error: ' + str(lossCurrent.item()), flush=True)
                        logger.info(str(epoch) + ' Training error: ' + str(lossCurrent.item()))
                        lr = optimizer_l.param_groups[0]["lr"]
                        print(f"Epoch {epoch} - Learning rate: {lr}")
                    epoch += 1
                return
        elif self.args['data'] == "CloudS2_128":
            dataset = CloudS2(idxs_train, self.args["path"], img_shape=(3, 128, 128))
            loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            
            if idxs_train_semisupervised is not None:
                dataset_semisupervised = CloudS2(idxs_train_semisupervised, self.args["path"], img_shape=(3, 128, 128))
                loader_tr_semisupervised = DataLoader(self.handler(dataset_semisupervised), shuffle=True, **self.args['loader_tr_args'])
            else:
                epoch = 0
                lossCurrent = 0.
                while epoch < n_epoch: 
                    lossCurrent = self._train(epoch, loader_tr, optimizer_l)
                    if epoch % 1 == 0:
                        print(str(epoch) + ' Training error: ' + str(lossCurrent.item()), flush=True)
                        logger.info(str(epoch) + ' Training error: ' + str(lossCurrent.item()))
                        lr = optimizer_l.param_groups[0]["lr"]
                        print(f"Epoch {epoch} - Learning rate: {lr}")
                    epoch += 1
                return
        elif self.args['data'] == "CloudL8_128":
            dataset = CloudL9(idxs_train, self.args["path"], img_shape=(3, 128, 128))
            loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            if idxs_train_semisupervised is not None:
                dataset_semisupervised = CloudL9(idxs_train_semisupervised, self.args["path"], img_shape=(3, 128, 128))
                loader_tr_semisupervised = DataLoader(self.handler(dataset_semisupervised), shuffle=True, **self.args['loader_tr_args'])
            else:
                epoch = 0
                lossCurrent = 0.
                while epoch < n_epoch: 
                    lossCurrent = self._train(epoch, loader_tr, optimizer_l)
                    if epoch % 1 == 0:
                        print(str(epoch) + ' Training error: ' + str(lossCurrent.item()), flush=True)
                        logger.info(str(epoch) + ' Training error: ' + str(lossCurrent.item()))
                        lr = optimizer_l.param_groups[0]["lr"]
                        print(f"Epoch {epoch} - Learning rate: {lr}")
                    epoch += 1
                return
        elif self.args['data'] == "CloudSEN12_128":
            dataset = CloudL9(idxs_train, self.args["path"], img_shape=(3, 128, 128))
            loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            if idxs_train_semisupervised is not None:
                dataset_semisupervised = CloudL9(idxs_train_semisupervised, self.args["path"], img_shape=(3, 128, 128))
                loader_tr_semisupervised = DataLoader(self.handler(dataset_semisupervised), shuffle=True, **self.args['loader_tr_args'])
            else:
                epoch = 0
                lossCurrent = 0.

                while epoch < n_epoch: 
                    lossCurrent = self._train(epoch, loader_tr, optimizer_l)
                    if epoch % 1 == 0:
                        print(str(epoch) + ' Training error: ' + str(lossCurrent.item()), flush=True)
                        logger.info(str(epoch) + ' Training error: ' + str(lossCurrent.item()))
                        lr = optimizer_l.param_groups[0]["lr"]
                        print(f"Epoch {epoch} - Learning rate: {lr}")
                    epoch += 1
                return
        elif self.args['data'] == "LandCoverAI_128":
            dataset = CloudL9(idxs_train, self.args["path"], img_shape=(3, 128, 128))
            loader_tr = DataLoader(self.handler(dataset), shuffle=True, **self.args['loader_tr_args'])
            if idxs_train_semisupervised is not None:
                dataset_semisupervised = CloudL9(idxs_train_semisupervised, self.args["path"], img_shape=(3, 128, 128))
                loader_tr_semisupervised = DataLoader(self.handler(dataset_semisupervised), shuffle=True, **self.args['loader_tr_args'])
            else:
                epoch = 0
                lossCurrent = 0.

                while epoch < n_epoch: 
                    lossCurrent = self._train(epoch, loader_tr, optimizer_l)
                    if epoch % 1 == 0:
                        print(str(epoch) + ' Training error: ' + str(lossCurrent.item()), flush=True)
                        logger.info(str(epoch) + ' Training error: ' + str(lossCurrent.item()))
                        lr = optimizer_l.param_groups[0]["lr"]
                        print(f"Epoch {epoch} - Learning rate: {lr}")
                    epoch += 1
                return
        else:
            raise NotImplementedError
        epoch = 0
        lossCurrent = 0.

        while epoch < n_epoch: 
            lossCurrent = self._train_with_semisupervised_learning(epoch, loader_tr, loader_tr_semisupervised, optimizer_l, optimizer_r, idxs_train, idxs_train_semisupervised)
            if epoch % 1 == 0:
                print(str(epoch) + f' Training error: {lossCurrent[0]}, Training loss branch1 {lossCurrent[1]}, Training loss branch2 {lossCurrent[2]}, Cross pseudo supervision loss {lossCurrent[3]}', flush=True)
                logger.info(str(epoch) + f' Training error: {lossCurrent[0]}, Training loss branch1 {lossCurrent[1]}, Training loss branch2 {lossCurrent[2]}, Cross pseudo supervision loss {lossCurrent[3]}')
                lr = optimizer_l.param_groups[0]["lr"]
                print(f"Epoch {epoch} - Learning rate: {lr}")
            epoch += 1
        
        
    def _train_with_semisupervised_learning(self, epoch, loader_tr, loader_tr_ss, optimizer_l, optimizer_r, idxs_train, idxs_train_semisupervised):
        
        self.reg.branch1.train()
        self.reg.branch2.train()
        
        # losses
        running_loss = 0
        running_loss_sup_l = 0
        running_loss_sup_r = 0
        running_cps_loss = 0
        all_idxs_sup = []
        all_idxs_unsup = []
        
        
        for batch_idx, data in enumerate(zip(loader_tr, loader_tr_ss)):
            
            supervised_data = data[0]
            unsupervised_data = data[1]
            
            x, y, idxs_sup = supervised_data
            x_ss, y_ss, idxs_unsup = unsupervised_data
            
            if len(y.shape) == 2:
                y = y.squeeze(-1)
            
            x, y = Variable(x.cuda()), Variable(y.cuda())
            x_ss, y_ss = Variable(x_ss.cuda()), Variable(y_ss.cuda())
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            
            if self.args['net_type'] in ["bayesian_net", "bayesian_resnet18", "bayesian_net_semi_supervised", "bayesian_resnet18_cloudscout"]:
                pred_sup_l = self.reg(x, k=1, step=1)
                pred_sup_l = pred_sup_l.squeeze(-1).squeeze(-1)
                
                pred_sup_r = self.reg(x, k=1, step=2)
                pred_sup_r = pred_sup_r.squeeze(-1).squeeze(-1)
                
                pred_unsup_l = self.reg(x_ss, k=1, step=1)
                pred_unsup_l = pred_unsup_l.squeeze(-1).squeeze(-1)
                
                pred_unsup_r = self.reg(x_ss, k=1, step=2)
                pred_unsup_r = pred_unsup_r.squeeze(-1).squeeze(-1)
                
                # Supervised losses
                loss_sup_l = torch.sqrt(F.mse_loss(pred_sup_l, y))      
                loss_sup_r = torch.sqrt(F.mse_loss(pred_sup_r, y))        
                
                # Semi-supervised loss
                if pred_sup_l.dim() == 0 or pred_sup_r.dim() == 0:
                    pred_sup_l = pred_sup_l.unsqueeze(0)
                    pred_sup_r= pred_sup_r.unsqueeze(0)
                    
                if pred_unsup_l.dim() == 0 or pred_unsup_r.dim() == 0:
                    pred_unsup_l = pred_unsup_l.unsqueeze(0)
                    pred_unsup_r = pred_unsup_r.unsqueeze(0)
                    
                pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
                pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
                
                cps_loss = torch.sqrt(F.mse_loss(pred_l, pred_r)) + torch.sqrt(F.mse_loss(pred_r, pred_l))
                
                # Total loss
                unsupervised_weight = 1
                loss = (loss_sup_l + loss_sup_r) + (unsupervised_weight * cps_loss)
                
            elif self.args['net_type'] in ["gaussian_net_semi_supervised"]:
                pred_sup_l = self.reg(x, step=1)
                mu_sup_l, sigma_sup_l = torch.chunk(pred_sup_l, 2, dim=-1)
                pred_sup_l = mu_sup_l.squeeze(-1).squeeze(-1)
                
                pred_sup_r = self.reg(x, step=2)
                mu_sup_r, sigma_sup_r = torch.chunk(pred_sup_r, 2, dim=-1)
                pred_sup_r = mu_sup_r.squeeze(-1).squeeze(-1)
                
                pred_unsup_l = self.reg(x_ss, step=1)
                mu_unsup_l, sigma_unsup_l = torch.chunk(pred_unsup_l, 2, dim=-1)
                pred_unsup_l = mu_unsup_l.squeeze(-1).squeeze(-1)
                
                pred_unsup_r = self.reg(x_ss, step=2)
                mu_unsup_r, sigma_unsup_r = torch.chunk(pred_unsup_r, 2, dim=-1)
                pred_unsup_r = mu_unsup_r.squeeze(-1).squeeze(-1)
                
                # Supervised losses
                loss_sup_l = torch.sqrt(F.mse_loss(pred_sup_l, y))
                loss_sup_r = torch.sqrt(F.mse_loss(pred_sup_r, y)) 
                
                loss_nll_sup_l = Gaussian_NLL(y, mu_sup_l, sigma_sup_l)
                loss_nll_sup_r = Gaussian_NLL(y, mu_sup_r, sigma_sup_r)
                
                # Semi-supervised loss
                if pred_sup_l.dim() == 0 or pred_sup_r.dim() == 0:
                    pred_sup_l = pred_sup_l.unsqueeze(0)
                    pred_sup_r= pred_sup_r.unsqueeze(0)
                    
                pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
                pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
                
                cps_loss = torch.sqrt(F.mse_loss(pred_l, pred_r)) + torch.sqrt(F.mse_loss(pred_r, pred_l))
                
                # Total loss
                unsupervised_weight = 1
                loss = (loss_sup_l + loss_sup_r) + (loss_nll_sup_l + loss_nll_sup_r) + (unsupervised_weight * cps_loss)
                
            elif self.args['net_type'] in ["evidential_net_semi_supervised", "evidential_math_regressor_SS"]:
                pred_sup_l = self.reg(x, step=1)
                mu_sup_l, v_sup_l, alpha_sup_l, beta_sup_l = torch.chunk(pred_sup_l, 4, dim=-1)
                mu_sup_l = mu_sup_l.squeeze(-1)
                v_sup_l = v_sup_l.squeeze(-1)
                alpha_sup_l = alpha_sup_l.squeeze(-1)
                beta_sup_l = beta_sup_l.squeeze(-1)
                
                pred_sup_r = self.reg(x, step=2)
                mu_sup_r, v_sup_r, alpha_sup_r, beta_sup_r = torch.chunk(pred_sup_r, 4, dim=-1)
                mu_sup_r = mu_sup_r.squeeze(-1)
                v_sup_r = v_sup_r.squeeze(-1)
                alpha_sup_r = alpha_sup_r.squeeze(-1)
                beta_sup_r = beta_sup_r.squeeze(-1)
                
                pred_unsup_l = self.reg(x_ss, step=1)
                mu_unsup_l, v_unsup_l, alpha_unsup_l, beta_unsup_l = torch.chunk(pred_unsup_l, 4, dim=-1)
                mu_unsup_l = mu_unsup_l.squeeze(-1)
                v_unsup_l = v_unsup_l.squeeze(-1)
                alpha_unsup_l = alpha_unsup_l.squeeze(-1)
                beta_unsup_l = beta_unsup_l.squeeze(-1)
                
                pred_unsup_r = self.reg(x_ss, step=2)
                mu_unsup_r, v_unsup_r, alpha_unsup_r, beta_unsup_r = torch.chunk(pred_unsup_r, 4, dim=-1)
                mu_unsup_r = mu_unsup_r.squeeze(-1)
                v_unsup_r = v_unsup_r.squeeze(-1)
                alpha_unsup_r = alpha_unsup_r.squeeze(-1)
                beta_unsup_r = beta_unsup_r.squeeze(-1)
                
                loss_sup_l = EvidentialRegressionLoss(y, pred_sup_l)
                loss_sup_r = EvidentialRegressionLoss(y, pred_sup_r)
                
                pred_l = torch.cat([mu_sup_l, mu_unsup_l], dim=0)
                pred_r = torch.cat([mu_sup_r, mu_unsup_r], dim=0)
                
                cps_loss = torch.sqrt(F.mse_loss(pred_l, pred_r)) + torch.sqrt(F.mse_loss(pred_r, pred_l))
                
                # Total loss
                unsupervised_weight = 1
                loss = (loss_sup_l + loss_sup_r) + (unsupervised_weight * cps_loss)
            elif self.args['net_type'] in ["beta_net_semi_supervised", "beta_math_regressor_SS", "beta_mobilenetv3_semi_supervised", "beta_mobilenetv4_semi_supervised"]:
                pred_sup_l = self.reg(x, step=1)
                mu_sup_l, nu_sup_l = torch.chunk(pred_sup_l, 2, dim=-1)
                mu_sup_l = mu_sup_l.squeeze(-1)
                nu_sup_l = nu_sup_l.squeeze(-1)
                
                pred_sup_r = self.reg(x, step=2)
                mu_sup_r, nu_sup_r = torch.chunk(pred_sup_r, 2, dim=-1)
                mu_sup_r = mu_sup_r.squeeze(-1)
                nu_sup_r = nu_sup_r.squeeze(-1)
                
                pred_unsup_l = self.reg(x_ss, step=1)
                mu_unsup_l, nu_unsup_l = torch.chunk(pred_unsup_l, 2, dim=-1)
                mu_unsup_l = mu_unsup_l.squeeze(-1)
                nu_unsup_l = nu_unsup_l.squeeze(-1)
                
                pred_unsup_r = self.reg(x_ss, step=2)
                mu_unsup_r, nu_unsup_r = torch.chunk(pred_unsup_r, 2, dim=-1)
                mu_unsup_r = mu_unsup_r.squeeze(-1)
                nu_unsup_r = nu_unsup_r.squeeze(-1)
                
                # Supervised losses
                # 1: Using only NLL
                # 2: Using only Reg
                # 3: Using both
                loss_sup_l = torch.sqrt(F.mse_loss(mu_sup_l, y))     
                loss_sup_r = torch.sqrt(F.mse_loss(mu_sup_r, y))

                loss_nll_sup_l = Beta_NLL(y, mu_sup_l, nu_sup_l)
                loss_nll_sup_r = Beta_NLL(y, mu_sup_r, nu_sup_r)
                
                # Semi-supervised loss
                if pred_sup_l.dim() == 0 or pred_sup_r.dim() == 0:
                    pred_sup_l = pred_sup_l.unsqueeze(0)
                    pred_sup_r= pred_sup_r.unsqueeze(0)
                    
                pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
                pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
                
                cps_loss = torch.sqrt(F.mse_loss(pred_l, pred_r)) + torch.sqrt(F.mse_loss(pred_r, pred_l))
                
                unsupervised_weight = 1
                nll_weight = 1
                reg_weight = 1
                
                loss = reg_weight*(loss_sup_l + loss_sup_r) + nll_weight*(loss_nll_sup_l + loss_nll_sup_r) + (unsupervised_weight * cps_loss)
            elif self.args['net_type'] in ["beta_net_semi_supervised_2"]:
                pred_sup_l = self.reg(x, step=1)
                y_hat_sup_l, mu_sup_l, nu_sup_l = torch.chunk(pred_sup_l, 3, dim=-1)
                y_hat_sup_l = y_hat_sup_l.squeeze(-1)
                mu_sup_l = mu_sup_l.squeeze(-1)
                nu_sup_l = nu_sup_l.squeeze(-1)
                
                pred_sup_r = self.reg(x, step=2)
                y_hat_sup_r, mu_sup_r, nu_sup_r = torch.chunk(pred_sup_r, 3, dim=-1)
                y_hat_sup_r = y_hat_sup_r.squeeze(-1)
                mu_sup_r = mu_sup_r.squeeze(-1)
                nu_sup_r = nu_sup_r.squeeze(-1)
                
                pred_unsup_l = self.reg(x_ss, step=1)
                y_hat_unsup_l, mu_unsup_l, nu_unsup_l = torch.chunk(pred_unsup_l, 3, dim=-1)
                y_hat_unsup_l = y_hat_unsup_l.squeeze(-1)
                mu_unsup_l = mu_unsup_l.squeeze(-1)
                nu_unsup_l = nu_unsup_l.squeeze(-1)
                
                pred_unsup_r = self.reg(x_ss, step=2)
                y_hat_unsup_r, mu_unsup_r, nu_unsup_r = torch.chunk(pred_unsup_r, 3, dim=-1)
                y_hat_unsup_r = y_hat_unsup_r.squeeze(-1)
                mu_unsup_r = mu_unsup_r.squeeze(-1)
                nu_unsup_r = nu_unsup_r.squeeze(-1)
                
                # Supervised losses
                loss_sup_l = torch.sqrt(F.mse_loss(y_hat_sup_l, y))
                loss_sup_r = torch.sqrt(F.mse_loss(y_hat_sup_r, y))

                loss_nll_sup_l = Beta_NLL(y, mu_sup_l, nu_sup_l)
                loss_nll_sup_r = Beta_NLL(y, mu_sup_r, nu_sup_r)
                
                if pred_sup_l.dim() == 0 or pred_sup_r.dim() == 0:
                    pred_sup_l = pred_sup_l.unsqueeze(0)
                    pred_sup_r= pred_sup_r.unsqueeze(0)
                    
                pred_l = torch.cat([y_hat_sup_l, y_hat_unsup_l], dim=0)
                pred_r = torch.cat([y_hat_sup_r, y_hat_unsup_r], dim=0)

                cps_loss = torch.sqrt(F.mse_loss(pred_l, pred_r)) + torch.sqrt(F.mse_loss(pred_r, pred_l))
                
                # Total loss
                unsupervised_weight = 1
                nll_weight = 1
                reg_weight = 0.5
                loss = reg_weight*(loss_sup_l + loss_sup_r) + nll_weight*(loss_nll_sup_l + loss_nll_sup_r) + (unsupervised_weight * cps_loss)    
            else:
                raise NotImplementedError
            
            running_loss += loss.detach().item()
            running_loss_sup_l += loss_sup_l.detach().item()
            running_loss_sup_r += loss_sup_r.detach().item()
            running_cps_loss += cps_loss.detach().item()
            loss.backward()

            optimizer_l.step()
            optimizer_r.step()
            
            all_idxs_sup.append(idxs_sup)
            all_idxs_unsup.append(idxs_unsup)
        
        # Supervised and unsupervised learning idxs
        all_idxs_sup = np.sort(np.concatenate(all_idxs_sup))
        all_idxs_unsup = np.sort(np.concatenate(all_idxs_unsup))
       
        epoch_loss = running_loss / len(loader_tr)
        epoch_loss_sup_l = running_loss_sup_l / len(loader_tr)
        epoch_loss_sup_r = running_loss_sup_r / len(loader_tr)
        epoch_cps_loss = running_cps_loss/len(loader_tr)
        self.training_info[f"epoch_{epoch}"] = {"loss": epoch_loss,
                                                "loss_sup_l": epoch_loss_sup_l,
                                                "loss_sup_r": epoch_loss_sup_r,
                                                "cps_loss": epoch_cps_loss,
                                                "all_idxs_sup": idxs_train[all_idxs_sup],
                                                "all_idxs_unsup": idxs_train_semisupervised[all_idxs_unsup],    
                                                }
    
        return epoch_loss, epoch_loss_sup_l, epoch_loss_sup_r, epoch_cps_loss   