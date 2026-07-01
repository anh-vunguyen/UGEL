# General packages
import numpy as np
import sys
import os
import argparse
import torch
import logging
import time
import pickle
import random
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import utils

# Dataset
from dataset import get_handler, CloudL9
from dataset_otf import DatasetOTF, DataHandlerOTF # On-the-fly data
from CloudL9_dataset import CloudL9
from CloudS2_dataset import CloudS2

# Sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# Models
# SS means the model type that can conduct Semi-Supervised Cross Pseudo Supervision
import models.resnet # Customed implementation
from models.resnet_family import ResNet50
from models.ssresnet import ResNet18_SS
from models.gaussian_resnet import gaussian_resnet18
from models.gaussian_ssresnet import Gaussian_Network_SS
from models.gaussian_ssresnet_v2 import Gaussian_Network_SS_v2
from models.evidential_resnet import evidential_resnet18
from models.evidential_ssresnet import Evidential_Network_SS
from models.bnn import BayesianNet
from models.bayesian_resnet import bayesian_resnet18
from models.beta_resnet import beta_resnet18
from models.beta_ssresnet import Beta_Network_SS
from models.beta_ssresnet_2 import Beta_Network_SS_2
from mobilenet.mobilenetv4 import MobileNetV4
from mobilenet.bayesian_mobilenetv4 import BayesianMobileNetV4
from mobilenet.beta_mobilenetv4 import BetaMobileNetV4
from mobilenet.beta_ssmobilenetv4 import Beta_MobileNetv4_SS

# Query algorithms
from query_strategies import RandomSampling, RandomSamplingOTF, BALD_reg, Uncertain_AL_Confident_SSL, Uncertain_AL_Confident_SSL_AllRemaining

def make_choices_help(choices):
   return '[' + ', '.join(choices) + ']'

# code based on https://github.com/ej0cl6/deep-active-learning"
parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--train_acc', help='training stopping criteria', type=float, default=0.99)
parser.add_argument('--zeta', help='z_t in equation 3', type=float, default=1)
parser.add_argument('--fill_random', default=True, action='store_true')
parser.add_argument('--single_pass', default=False, action='store_true')
parser.add_argument('--deterministic', default=False, action='store_true', help="in streaming random sampler, whether to select every kth samples deterministically or select randomly with fixed rate")
parser.add_argument('--embs', default="grad_embs", help="whether to use gradient embeddings (grad_embs) or penultimate layer embeddings (penultimate).", type=str)
parser.add_argument('--stream_sampler_early_stop', default=False, action='store_true')
parser.add_argument('--model', help='model - resnet or mlp or cloudscout', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset' , type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
parser.add_argument('--nStart', help='number of points to start', type=int, default=0)
parser.add_argument('--nEnd', help = 'total number of points to query', type=int, default=50000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)
parser.add_argument('--rank', help='rank of the sample-wise fisher information matrix', type=int, default=1)
parser.add_argument('--cov_inv_scaling', help='covariance inverse scaling', type=float, default=100)
parser.add_argument('--activation', help=make_choices_help(['square','relu','sigmoid']), type=str, default='square', metavar='activation function for gradient descent autotuning')
parser.add_argument('--use_load_otf', help=make_choices_help(["True", "False"]), type=str, default='False', metavar='whether to use load-on-the-fly DataLoader or load-entire-dataset-at-once DataLoader')
parser.add_argument('--use_two_stage_training', help='using two stage training for cloud detection model (only for Cloud9 dataset)', type=str, default='False')
parser.add_argument('--initial_set', help="The initial training set", type=int, default=None)
parser.add_argument('--use_same_init_weights', help='using the same initialized weights for all rounds', default='False')
parser.add_argument('--train_path', help='train path', default='')
parser.add_argument('--test_path', help='test path', default='')
parser.add_argument('--input_dim', help='input dimension', type=int, default=9)
opts = parser.parse_args()

if opts.use_load_otf == "True":
    opts.use_load_otf = True
else:
    opts.use_load_otf = False

# Two-stage training (CloudScout)
if opts.data == 'CloudL9' and opts.model == "cloudscout" and opts.use_two_stage_training == "True":
    opts.use_two_stage_training = True
else:
    opts.use_two_stage_training = False
    
if opts.use_same_init_weights == "True":
    opts.use_same_init_weights = True
else:
    opts.use_same_init_weights = False
    
# Start time for the entire experiment
start_time_org = time.time()

# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB)/ opts.nQuery)
DATA_NAME = opts.data

args_pool = {'MNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'SVHN':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'CIFAR10': # NOTE: Jun 02 - Current version: ImbStr_CIFAR2 (selected class 0 / 1)
                # n_epoch: 3
                {'n_epoch': 100, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.3},
                 'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])},
            # Our Synthetic Dataset
            # NOTE: The values are just placeholder values.
            # TODO: Change the parameter values
            'CloudL9':
                {'n_epoch': 30, #30, # 300, #300, 
                 'transform': None,
                 'loader_tr_args':{'batch_size': 6, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 6, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.3}, # Old: 0.01
                 'transformTest': None},
            'CloudS2':
                {'n_epoch': 30, #30, # 300, #300, 
                 'transform': None,
                 'loader_tr_args':{'batch_size': 6, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 6, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.3}, # Old: 0.01
                 'transformTest': None},
            'CloudL9_128':
                {'n_epoch': 12, # NOTE: 14/04/2025: previous 36
                 'transform': None,
                 'loader_tr_args':{'batch_size': 6, 'num_workers': 1}, # old: 6
                 'loader_te_args':{'batch_size': 256, 'num_workers': 1}, # old: 6
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.3}, # Old: 0.01
                 'transformTest': None},    
            'CloudS2_128':
                {'n_epoch': 36, 
                 'transform': None,
                 'loader_tr_args':{'batch_size': 6, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 6, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.3}, # Old: 0.01
                 'transformTest': None},
            'CloudL8_128':
                {'n_epoch': 12, # 36
                 'transform': None,
                 'loader_tr_args':{'batch_size': 6, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 256, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.3}, # Old: 0.01
                 'transformTest': None},
            'CloudSEN12_128': # NOTE: 26/06/2025
                {'n_epoch': 12, # 36
                 'transform': None,
                 'loader_tr_args':{'batch_size': 6, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 256, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.3}, # Old: 0.01
                #  'optimizer_args':{'lr': 0.01, 'momentum': 0.3}, # NOTE: 08/10/2025 AVN TEST
                 
                 'transformTest': None},
            'LandCoverAI_128': # NOTE: 26/06/2025
                {'n_epoch': 12, # 36
                 'transform': None,
                 'loader_tr_args':{'batch_size': 6, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 256, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.3}, # Old: 0.01
                 'transformTest': None},
            }

# Number of classes
if opts.data in ['CloudL9', 'CloudL9_128', 'CloudS2_128', 'CloudL8_128', 'CloudSEN12_128', 'LandCoverAI_128']:
    opts.nClasses = 1
elif opts.data[:3] == "uci":
    opts.nClasses = 1
else:
    print("BE CAREFUL HERE!, CIFAR2 IS BEING USED") # CIFAR with 2 classes, 
    opts.nClasses = 2 # 2 
    
args_pool['CIFAR10']['transform'] =  args_pool['CIFAR10']['transformTest'] # remove data augmentation
args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']

if not os.path.exists(opts.path):
    os.makedirs(opts.path)

if not opts.use_load_otf:
    if opts.data == 'CloudL9':  # RSRC-L9
        cloud_threshold = 0.7 # obsoleted because we pivoted to regression
        train_path = opts.train_path
        print(f"Train Path: {train_path}")
        train_ids = np.arange(12000)
        print(train_ids)
        test_path = opts.test_path 
        print(f"Test Path: {test_path}")
        test_ids = np.arange(2000)
        print(test_ids)
        handler = get_handler(opts.data)
        cloud_train = CloudL9(train_ids, train_path, task="regression", two_stage_training=opts.use_two_stage_training, cloud_threshold=cloud_threshold) 
        cloud_test = CloudL9(test_ids, test_path, task="regression", cloud_threshold=cloud_threshold) 
        n_pool = len(cloud_train)
        n_test = len(cloud_test)
    elif opts.data == 'CloudS2': # RSRC-S2
        assert os.path.exists(opts.path), f"{opts.path} does not exist."
        file_list = np.array(os.listdir(opts.path))
        cloud_threshold = 0.7
        train_ids = np.arange(20000)
        print(train_ids)
        test_ids = np.arange(20000, 30000)
        print(test_ids)
        handler = get_handler(opts.data)
        cloud_train = CloudS2(train_ids, opts.path, two_stage_training=opts.use_two_stage_training, cloud_threshold=cloud_threshold) 
        cloud_test = CloudS2(test_ids, opts.path, cloud_threshold=cloud_threshold) 
        n_pool = len(cloud_train)
        n_test = len(cloud_test)
    elif opts.data == "CloudL9_128": # 128 x 128``
        assert os.path.exists(opts.path), f"{opts.path} does not exist."
        file_list = np.array(os.listdir(opts.path))
        cloud_threshold = 0.7
        # all_idxs = np.load('/data/betaU_train.npy')
        # all_idxs = np.load('/data/negative_skew_train.npy')
        # all_idxs = np.load('/data/positive_skew_train.npy')
        # all_idxs = np.load('/data/uniform_train.npy')
        # all_idxs = np.load('/data/symmetric_train.npy')
        all_idxs = np.load('/data/normal_train_sigma015.npy')
        # all_idxs = np.load('/data/normal_train_loc03_sigma015.npy')
        nb_train = len(all_idxs)
        np.random.shuffle(all_idxs)
        train_ids = all_idxs[:nb_train]
        print(f"Training set: {sorted(train_ids)}")
        # Initial set
        idxs_lb = np.zeros(nb_train, dtype=bool)
        idxs_lb[:NUM_INIT_LB] = True # First 100 frames
        print(train_ids[:NUM_INIT_LB])
        
        print("Independent testing")
        # test_ids = np.load('/data/betaU_test.npy')
        # test_ids = np.load('/data/negative_skew_test.npy')
        # test_ids = np.load('/data/positive_skew_test.npy')
        # test_ids = np.load('/data/uniform_test.npy')
        # test_ids = np.load('/data/symmetric_test.npy')
        test_ids = np.load('./data/normal_test_sigma015.npy')
        # test_ids = np.load('/data/normal_test_loc03_sigma015.npy')
        cloud_test_path = "/hpcfs/users/a1872455/data/CloudL9_ind_test128"
        handler = get_handler(opts.data)
        cloud_train = CloudL9(train_ids, opts.path, img_shape=(3, 128, 128), two_stage_training=opts.use_two_stage_training, cloud_threshold=cloud_threshold) 
        cloud_test = CloudL9(test_ids, cloud_test_path, img_shape=(3, 128, 128), cloud_threshold=cloud_threshold) 
        n_pool = len(cloud_train)
        n_test = len(cloud_test)
    elif opts.data == "CloudS2_128":
        assert os.path.exists(opts.path), f"{opts.path} does not exist."
        assert os.path.exists(opts.test_path), f"{opts.test_path} does not exist."
        
        file_list = np.array(os.listdir(opts.path))
        cloud_threshold = 0.7
        all_idxs = np.arange(16000)
        nb_train = 16000
        nb_test = 8000
        
        # np.random.seed(9) # NOTE: SET SEED
        
        np.random.shuffle(all_idxs) # Turn into i.i.d
        train_ids = all_idxs[:nb_train]
        print(f"Training set: {sorted(train_ids)}")
        
        # Initial set
        idxs_lb = np.zeros(nb_train, dtype=bool)
        idxs_lb[:NUM_INIT_LB] = True # First 100 frames
        print(train_ids[:NUM_INIT_LB])
        cloud_train = CloudS2(train_ids, opts.path, img_shape=(3, 128, 128), two_stage_training=opts.use_two_stage_training, cloud_threshold=cloud_threshold) 
        
        print("Independent testing")
        test_ids = np.arange(8000)
        cloud_test_path = opts.test_path
        
        cloud_test = CloudS2(test_ids, cloud_test_path, img_shape=(3, 128, 128), cloud_threshold=cloud_threshold) 
        handler = get_handler(opts.data)
        n_pool = len(cloud_train)
        n_test = len(cloud_test)
    elif opts.data == "CloudL8_128":
        assert os.path.exists(opts.path), f"{opts.path} does not exist."
        assert os.path.exists(opts.test_path), f"{opts.test_path} does not exist."
        file_list = np.array(os.listdir(opts.path))
        cloud_threshold = 0.7
        all_idxs = np.arange(20294)
        np.random.shuffle(all_idxs)
        
        nb_train = 20294  # data128_57
        nb_test = 22528  # data128_38
        
        train_ids = all_idxs
        
        # Initial set
        idxs_lb = np.zeros(nb_train, dtype=bool)
        idxs_lb[:NUM_INIT_LB] = True # First 100 frames
        print(train_ids[:NUM_INIT_LB])
        
        test_ids = np.arange(nb_test)
        test_path = opts.test_path
        
        handler = get_handler(opts.data)
        cloud_train = CloudL9(train_ids, opts.path, img_shape=(3, 128, 128), two_stage_training=opts.use_two_stage_training, cloud_threshold=cloud_threshold) 
        cloud_test = CloudL9(test_ids, test_path, img_shape=(3, 128, 128), cloud_threshold=cloud_threshold) 
        n_pool = len(cloud_train)
    elif opts.data == "CloudSEN12_128":
        assert os.path.exists(opts.path), f"{opts.path} does not exist."
        assert os.path.exists(opts.test_path), f"{opts.test_path} does not exist."
        file_list = np.array(os.listdir(opts.path))
        cloud_threshold = 0.7 # Not in used
        all_idxs = np.arange(80000)
        np.random.shuffle(all_idxs)
        nb_train = 80000  # data128_57
        nb_test = 32000   # data128_38
        train_ids = all_idxs
        
        # Initial set
        idxs_lb = np.zeros(nb_train, dtype=bool)
        idxs_lb[:NUM_INIT_LB] = True # First 100 framv
        print(train_ids[:NUM_INIT_LB])
        
        test_ids = np.arange(nb_test)
        test_path = opts.test_path

        handler = get_handler(opts.data)
        cloud_train = CloudL9(train_ids, opts.path, img_shape=(3, 128, 128), two_stage_training=opts.use_two_stage_training, cloud_threshold=cloud_threshold) 
        cloud_test = CloudL9(test_ids, test_path, img_shape=(3, 128, 128), cloud_threshold=cloud_threshold) 
        n_pool = len(cloud_train)
        n_test = len(cloud_test)
    elif opts.data == "LandCoverAI_128":
        assert os.path.exists(opts.path), f"{opts.path} does not exist."
        assert os.path.exists(opts.test_path), f"{opts.test_path} does not exist."
        file_list = np.array(os.listdir(opts.path))
        cloud_threshold = 0.7 # Not in used
        
        all_idxs = np.arange(154056) 
        np.random.shuffle(all_idxs)
        
        
        nb_train = 154056
        nb_test = 25279
        
        train_ids = all_idxs 
        
        
        # Initial set
        idxs_lb = np.zeros(nb_train, dtype=bool)
        idxs_lb[:NUM_INIT_LB] = True # First 100 framv
        print(train_ids[:NUM_INIT_LB])
        
        test_ids = np.arange(nb_test)      
        test_path = opts.test_path
        
        
        handler = get_handler(opts.data)
        cloud_train = CloudL9(train_ids, opts.path, img_shape=(3, 128, 128), two_stage_training=opts.use_two_stage_training, cloud_threshold=cloud_threshold) 
        cloud_test = CloudL9(test_ids, test_path, img_shape=(3, 128, 128), cloud_threshold=cloud_threshold) 
        n_pool = len(cloud_train)
        n_test = len(cloud_test)
    else:
        raise NotImplementedError
        
args = args_pool[DATA_NAME]
args['path'] = opts.path
args['nClasses'] = opts.nClasses # Updated Jun 18
args['lr'] = opts.lr
args['train_acc'] = opts.train_acc

# Start experiment
print('Number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('Number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('Number of testing pool: {}'.format(n_test), flush=True)

# Generate initial labeled pool
idxs_tmp = np.arange(n_pool)

args['initial_set'] = opts.initial_set
if opts.initial_set is None:
    pass
else:
    # We do not set fixed initial set.
    raise NotImplementedError

# Models and Customed models
class mlpMod(nn.Module):
    def __init__(self, dim, embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, opts.nClasses)
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb
    def get_embedding_dim(self):
        return self.embSize

class pretrained(nn.Module):
     def __init__(self, nClasses=40):
        # from torchvision.models import resnet18, ResNet18_Weights
        super(pretrained, self).__init__()
        self.backbone = torch.load('pretrained_resnet.pt')
        self.backbone.fc = nn.Identity(512, 512)
        self.lin = nn.Linear(512, nClasses)
     def forward(self, x):
        emb = self.backbone(x)
        out = self.lin(emb)
        return out, emb

     def get_embedding_dim(self):
        return 512
    
# Pretrained ResNet18 - Finetuning approach
class pretrained_resnet18(nn.Module):
     def __init__(self, nClasses=1):
        # from torchvision.models import resnet18, ResNet18_Weights
        super(pretrained_resnet18, self).__init__()
        # self.backbone = resnet18(weights=ResNet18_Weights)
        self.backbone = resnet18()
        pretrained_checkpoint = torch.load('./checkpoints/resnet18-f37072fd.pth')
        self.backbone.load_state_dict(pretrained_checkpoint)
        self.backbone.fc = nn.Identity(512, 512)
        
        # Freeze the backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        self.lin = nn.Linear(512, nClasses)
     def forward(self, x):
        emb = self.backbone(x)
        out = self.lin(emb)
        out = torch.sigmoid(out)
        return out

     def get_embedding_dim(self):
        return 512

# ResNet18-based regressor
class resnet18_(nn.Module):
     def __init__(self, nClasses=2):
        # from torchvision.models import resnet18, ResNet18_Weights
        super(resnet18_, self).__init__()
        self.backbone = resnet18()
        self.backbone.fc = nn.Identity(512, 512)        
        self.lin = nn.Linear(512, nClasses)
        
     def forward(self, x):
        emb = self.backbone(x)
        out = self.lin(emb)
        out = torch.sigmoid(out) # NOTE: 13 January 2025
        # return out, emb
        return out

# Bayesian Resnet
class bayesian_resnet18_(nn.Module):
     def __init__(self, num_classes=1):
        # from torchvision.models import resnet18, ResNet18_Weights
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

     def get_embedding_dim(self):
        return 512


class bayesian_mobilenetv3_(nn.Module):
     def __init__(self, num_classes=1):
        # from torchvision.models import resnet18, ResNet18_Weights
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
        

class bayesian_mobilenetv4_(nn.Module):
     def __init__(self, num_classes=1):
        # from torchvision.models import resnet18, ResNet18_Weights
        super(bayesian_mobilenetv4_, self).__init__()
        self.net = BayesianMobileNetV4("MobileNetV4ConvSmall")
        self.k = 10
        
     def forward(self, x, k=1):
        if k == 1:
            out = torch.sigmoid(self.net(x))
            return out
        else:
            # self.net.enable_mc_dropout()
            outs = []
            for i in range(k):
                out = torch.sigmoid(self.net(x))
                outs.append(out.squeeze(-1))
            outs = torch.vstack(outs)
            outs = outs.permute(1, 0)
            return outs
        
        
class convolutional_regressor_a(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5,)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # NOTE: 02 January 2025
        if x.dim() == 5:
            x = x.squeeze(1)
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        
        
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    
def save_checkpoint(net, path, savedname):
    saved_path = os.path.join(path, savedname)
    torch.save({'model_state_dict': net.state_dict()}, saved_path)
    logger.info(f"Saved {savedname}")
    
def save_training_info(strategy_otf, path, savedname):
    saved_path = os.path.join(path, savedname)
    with open(saved_path, 'wb') as f:
        pickle.dump(strategy_otf.training_info, f)
        
# Load specified network
if opts.model == 'mlp':
    net = mlpMod(opts.dim, embSize=opts.nEmb)
elif opts.model == 'pretrained':
    net =  pretrained(nClasses=opts.nClasses)
elif opts.model == 'resnet':
    net = models.resnet.ResNet18(num_classes=opts.nClasses)
elif opts.model == 'pretrained_resnet18': # NOTE: Pretrained Resnet 18, Finetuning
    net = pretrained_resnet18(nClasses=opts.nClasses)
elif opts.model == "resnet18": # ResNet18 and using same initalisation for all AL rounds
    net = resnet18_(nClasses=opts.nClasses)
elif opts.model == "resnet18_semi_supervised": # NOTE: 24 July 2025
    net = ResNet18_SS(num_classes=1)
elif opts.model == "pretrainedL9_resnet50":
    # Note: 24 July 2024
    # Regression
    net  = ResNet50(num_classes=1, num_bands=3)
    # print('Turn off pretrained checkpoint')
    # NOTE: 29/07/2024 Turn of pretrained checkpoint
    # NOTE: 24/10/2024 New checkpoint
    # checkpoint = torch.load('/gpfs/users/a1872455/SAL_December/new_experimental_design/SAL_Cloud9_reg/checkpoints/resnet50-432-L9-2024/saves-train/regression_model70-epoch-30.ckpt', map_location=torch.device('cuda'))
    # checkpoint = torch.load('/home/anhvunguyen/workspace/workspace_AV/2024/SAL_November/SAL_Cloud9_reg/checkpoints/resnet50-432-L9-2024/saves-train/regression_model70-epoch-30.ckpt', map_location=torch.device('cuda'))
    # net.load_state_dict(checkpoint)
    
elif opts.model == "pretrainedS2_resnet50":
    # Regression
    net  = ResNet50(num_classes=1, num_bands=3)
    checkpoint = torch.load('./checkpoints/resnet50-432-S2-2024/saves-train/regression_model70-epoch-30.ckpt', map_location=torch.device('cuda'))
    # checkpoint = torch.load('/home/anhvunguyen/workspace/workspace_AV/2024/SAL_November/SAL_Cloud9_reg/checkpoints/resnet50-432-S2-2024/saves-train/regression_model70-epoch-30.ckpt', map_location=torch.device('cuda'))
    net.load_state_dict(checkpoint)
elif opts.model == "vanilla_resnet50":
    net  = ResNet50(num_classes=1, num_bands=3)
elif opts.model == "convolutional_regressor_a": # NOTE: 13 January 2025
    net = convolutional_regressor_a()
elif opts.model == "bayesian_net":
    net = BayesianNet(num_classes=1)
elif opts.model == "bayesian_resnet18":
    net = bayesian_resnet18_(num_classes=1)
elif  opts.model == "bayesian_net_semi_supervised":
    net = Bayesian_Network_SS(num_classes=1) # NOTE: 27/01/2025
elif opts.model == "bayesian_resnet18_cloudscout": # NOTE:  10/02/2025
    net = Bayesian_ResNet_CloudScout(num_classes=1, cloudscout_checkpoint="/gpfs/users/a1872455/2025/active_semi_reg/cloudscout_reg_epoch30.ckpt")
elif opts.model == "gaussian_resnet18":
    net = gaussian_resnet18(num_classes=1)
elif opts.model == "gaussian_net_semi_supervised":
    net = Gaussian_Network_SS(num_classes=1)
elif opts.model == "gaussian_net_semi_supervised_v2": # NOTE: 22/04/2025
    net = Gaussian_Network_SS_v2(num_classes=1) # NOTE: 22/04/2025
elif opts.model == "evidential_resnet18":    
    net = evidential_resnet18(num_classes=1)
elif opts.model == "evidential_net_semi_supervised":
    net = Evidential_Network_SS(num_classes=1)
elif opts.model == "beta_resnet18":
    net = beta_resnet18(num_classes=1)
elif opts.model == "beta_net_semi_supervised":
    net = Beta_Network_SS(num_classes=1)
elif opts.model == "beta_net_semi_supervised_2": # NOTE: 22/05/2025
    net = Beta_Network_SS_2(num_classes=1) # NOTE: 22/05/2025 
elif opts.model == "math_regressor": # NOTE: 05/05/2025
    net = math_regressor(input_dim=opts.input_dim)
elif opts.model == "mcdropout_math_regressor":
    net = mcdropout_math_regressor_(input_dim=opts.input_dim)
elif opts.model == "mcdropout_math_regressor_SS":
    net = mcdropout_math_regressor_SS(input_dim=opts.input_dim)
elif opts.model == "evidential_math_regressor":
    net = evidential_math_regressor(input_dim=opts.input_dim)
elif opts.model == "evidential_math_regressor_SS":
    net = evidential_math_regressor_SS(input_dim=opts.input_dim)
elif opts.model == "beta_math_regressor":
    net = beta_math_regressor(input_dim=opts.input_dim)
elif opts.model == "beta_math_regressor_SS":
    net = beta_math_regressor_SS(input_dim=opts.input_dim)
elif opts.model == "mobilenetv3": # 08/10/2025
    net = mobilenetv3_small()
elif opts.model == "bayesian_mobilenetv3": # 08/10/2025
    net = bayesian_mobilenetv3_()
elif opts.model == "beta_mobilenetv3": # 09/10/2025
    net = beta_mobilenetv3_small()
elif opts.model == "beta_mobilenetv3_semi_supervised": # 09/10/2025
    net = Beta_MobileNetv3_SS()
elif opts.model == "mobilenetv4": # 10/10/2025
    net = MobileNetV4("MobileNetV4ConvSmall")
elif opts.model == "bayesian_mobilenetv4": # 08/10/2025
    net = bayesian_mobilenetv4_()
elif opts.model == "beta_mobilenetv4":
    net = BetaMobileNetV4("MobileNetV4ConvSmall")
elif opts.model == "beta_mobilenetv4_semi_supervised":
    net = Beta_MobileNetv4_SS() 
else: 
    print('choose a valid model - mlp, resnet, or pretrained', flush=True)
    raise ValueError

args["zeta"] = float(opts.zeta)
args["fill_random"] = opts.fill_random
args["early_stop"] = opts.stream_sampler_early_stop
args["single_pass"] = opts.single_pass
args['rank'] = opts.rank
args['cov_inv_scaling'] = opts.cov_inv_scaling
args['data'] = opts.data
args['activation'] = opts.activation
args['embs'] = opts.embs
args["deterministic"] = opts.deterministic
args['use_same_init_weights'] = opts.use_same_init_weights

if opts.alg == 'rand_otf': # random sampling (baseline)
    strategy = RandomSamplingOTF(cloud_train, idxs_lb, net, handler, args)
elif  opts.alg == "bald_reg":
    strategy = BALD_reg(cloud_train, idxs_lb, net, handler, args) # Jan 17 2024
elif opts.alg == "uncertain_al_confident_ssl":
    strategy = Uncertain_AL_Confident_SSL(cloud_train, idxs_lb, net, handler, args)
elif opts.alg == "uncertain_al_confident_ssl_all_remaining":
    strategy = Uncertain_AL_Confident_SSL_AllRemaining(cloud_train, idxs_lb, net, handler, args)
elif opts.alg == "only_ssl":
    strategy = Only_SSL(cloud_train, idxs_lb, net, handler, args)
else: 
        print('choose a valid acquisition function', flush=True)
        raise ValueError
    
get_pretrained = lambda : pretrained(nClasses=opts.nClasses)
args['pretrained_resnet'] = get_pretrained

get_pretrained_resnet18 = lambda : pretrained_resnet18(nClasses=opts.nClasses)
args['pretrained_resnet18'] = get_pretrained_resnet18
args['net_type'] = opts.model

get_bayesian_resnet18 = lambda : bayesian_resnet18_(num_classes=1)
args['bayesian_resnet18'] = get_bayesian_resnet18

get_bayesian_net_semi_supervised = lambda : Bayesian_Network_SS(num_classes=1)
args['bayesian_net_semi_supervised'] = get_bayesian_net_semi_supervised

get_bayesian_resnet18_cloudscout = lambda: Bayesian_ResNet_CloudScout(num_classes=1, cloudscout_checkpoint="/gpfs/users/a1872455/2025/active_semi_reg/cloudscout_reg_epoch30.ckpt")
args['bayesian_resnet18_cloudscout'] = get_bayesian_resnet18_cloudscout

get_gaussian_resnet18 = lambda : gaussian_resnet18(num_classes=1)
args['gaussian_resnet18'] = get_gaussian_resnet18

get_gaussian_net_semi_supervised = lambda : Gaussian_Network_SS(num_classes=1)
args['gaussian_net_semi_supervised'] = get_gaussian_net_semi_supervised

get_evidential_resnet18 = lambda : evidential_resnet18(num_classes=1)
args['evidential_resnet18'] = get_evidential_resnet18

get_evidential_net_semi_supervised = lambda : Evidential_Network_SS(num_classes=1)
args['evidential_net_semi_supervised'] = get_evidential_net_semi_supervised

get_beta_resnet18 = lambda : beta_resnet18(num_classes=1)
args['beta_resnet18'] = get_beta_resnet18

get_beta_net_semi_supervised = lambda : Beta_Network_SS(num_classes=1)
args['beta_net_semi_supervised'] = get_beta_net_semi_supervised

get_beta_net_semi_supervised_2 = lambda : Beta_Network_SS_2(num_classes=1)
args['beta_net_semi_supervised_2'] = get_beta_net_semi_supervised_2

get_resnet18_semi_supervised = lambda : ResNet18_SS(num_classes=1)
args['resnet18_semi_supervised'] = get_resnet18_semi_supervised

get_bayesian_mobilenetv3 = lambda : bayesian_mobilenetv3_()
args['bayesian_mobilenetv3'] = get_bayesian_mobilenetv3

get_mobilenetv3 = lambda : mobilenetv3_small()
args['mobilenetv3'] = get_mobilenetv3

get_beta_mobilenetv3 = lambda : beta_mobilenetv3_small()
args['beta_mobilenetv3'] = get_beta_mobilenetv3

get_beta_mobilenetv3_semi_supervised = lambda : Beta_MobileNetv3_SS()
args['beta_mobilenetv3_semi_supervised'] = get_beta_mobilenetv3_semi_supervised

# 13 November 2025
get_mobilenetv4 = lambda : MobileNetV4("MobileNetV4ConvSmall")
args['mobilenetv4'] = get_mobilenetv4

get_bayesian_mobilenetv4 = lambda : bayesian_mobilenetv4_()
args['bayesian_mobilenetv4'] = get_bayesian_mobilenetv4

get_beta_mobilenetv4 = lambda : BetaMobileNetV4("MobileNetV4ConvSmall")
args['beta_mobilenetv4'] = get_beta_mobilenetv4

get_beta_mobilenetv4_semi_supervised = lambda : Beta_MobileNetv4_SS()
args['beta_mobilenetv4_semi_supervised'] = get_beta_mobilenetv4_semi_supervised

print(DATA_NAME, flush=True)
print(type(strategy).__name__, flush=True)
WORK_DIRS = './work_dirs/'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging_dir_name = opts.alg + "_" + opts.data + '_' + opts.model # Algorithm + Dataset + Net
full_logging_dir_name = WORK_DIRS + logging_dir_name + f"/{timestamp}" + "_" + str(random.randint(0, 9999))
utils.mkdir_if_not_exist(full_logging_dir_name)
args['full_logging_dir_name'] = full_logging_dir_name
print(f"Created logging directory: {full_logging_dir_name}")

logfile = os.path.join(full_logging_dir_name, f'{timestamp}.log')
resultfile = os.path.join(full_logging_dir_name, f'{timestamp}.pkl')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

ch = logging.FileHandler(logfile)
formatter = logging.Formatter(fmt='%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

env_info_dict = utils.collect_env()
env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
opts_info = '\n'.join([(f'{k}: {v}') for k, v in vars(opts).items()])
logger.info('Options:\n' + dash_line + opts_info + '\n' + dash_line)
args_info = '\n'.join([(f'{k}: {v}') for k, v in args.items()])
logger.info('Arguments:\n' + dash_line + opts_info + '\n' + dash_line)
print('Number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
logger.info('Number of labeled pool: {}'.format(NUM_INIT_LB))
print('Number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
logger.info('Number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('Number of testing pool: {}'.format(n_test), flush=True)
logger.info('Number of testing pool: {}'.format(n_test))
logger.info(dash_line)

np.save(os.path.join(full_logging_dir_name, 'initial_set'), idxs_tmp[:NUM_INIT_LB])

# NOTE: if we use same init weights
if opts.use_same_init_weights:
    save_checkpoint(strategy.reg, full_logging_dir_name, f"init_weights.pt")
    
# # Round 0 accuracy
logger.info('Round 0')
start_time = time.time()
if opts.use_two_stage_training:
    strategy.train_two_stage() # For CloudScout
else:
    strategy.train()

if opts.use_same_init_weights and opts.alg == 'pretrained_based_otf':
    print("Saved finetuned weights")
    save_checkpoint(net, full_logging_dir_name, f"init_weights.pt")

Y_te = cloud_test.get_target(use_threshold=False)
cloud_test.current_stream_id = -1
P = strategy.predict(cloud_test)

nb_training_samples = sum(idxs_lb)
output_str = str(nb_training_samples) + '\t' + 'Testing error RMSE {}'.format(torch.sqrt(F.mse_loss(P, Y_te)))
print(output_str, flush=True)
logger.info(output_str)
output_str = str(nb_training_samples) + '\t' + 'Testing error MSE {}'.format(F.mse_loss(P, Y_te))
print(output_str, flush=True)
logger.info(output_str)
output_str = str(nb_training_samples) + '\t' + 'Testing error MAE {}'.format(F.l1_loss(P, Y_te))
print(output_str, flush=True)
logger.info(output_str)

scan_per_round = (len(cloud_train) - NUM_INIT_LB) // NUM_ROUND

strategy.allowed = np.zeros(strategy.n_pool, dtype=bool)
for rd in range(1, NUM_ROUND+1):
    if opts.single_pass:
        strategy.allowed[rd*scan_per_round: (rd+1)*scan_per_round] = True 
    print('Round {}'.format(rd), flush=True)
    logger.info('Round {}'.format(rd))
    start_time = time.time()
    # query
    output = strategy.query(NUM_QUERY)#, rd)
    output = np.sort(output)
    q_idxs = output
    if len(q_idxs) == 0:
        break
    idxs_lb[q_idxs] = True
    # ===================
    
    # if output is not None:
    #     # NOTE: Sorting
    #     output = np.sort(output)
    #     q_idxs = output
    #     if len(q_idxs) == 0:
    #         break
    #     idxs_lb[q_idxs] = True
    #     # Update
    #     strategy.update(idxs_lb)
    
    # Update
    strategy.update(idxs_lb)
    
    # NOTE: Make sure training loss is minimized
    if opts.use_two_stage_training:
        strategy.train_two_stage()
    else:
        strategy.train()

    # Saving losses and training accuracy
    save_training_info(strategy, full_logging_dir_name, f"{opts.model}_rd{str(rd).zfill(3)}.pkl")
    # Saving checkpopint
    if rd % 10 == 0:
        save_checkpoint(strategy.reg, full_logging_dir_name, f"{opts.model}_rd{str(rd).zfill(3)}.pt")

    cloud_test.current_stream_id = -1
    P = strategy.predict(cloud_test)
    
    nb_training_samples = sum(idxs_lb)
    output_str = str(nb_training_samples) + '\t' + 'Testing error RMSE {}'.format(torch.sqrt(F.mse_loss(P, Y_te)))
    print(output_str, flush=True)
    logger.info(output_str)
    output_str = str(nb_training_samples) + '\t' + 'Testing error MSE {}'.format(F.mse_loss(P, Y_te))
    print(output_str, flush=True)
    logger.info(output_str)
    output_str = str(nb_training_samples) + '\t' + 'Testing error MAE {}'.format(F.l1_loss(P, Y_te))
    print(output_str, flush=True)
    logger.info(output_str)
    logger.info(f"Total time: {time.time() - start_time}s")
    
logger.info(f'Total time for experiment: {time.time() - start_time_org}')
