"""Cloud9: Landsat 9's satellite rgb [band 4-3-2] images augmented with synthetic clouds.
Authors: Anh Vu Nguyen
"""
import os
import pandas as pd
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import time
import logging
import gc

class CloudL9(Dataset):
    def __init__(self, 
                 selected_ids, 
                 data_dir, 
                 task="regression", 
                 img_shape=(3, 128, 128), 
                 cloud_threshold=0.7, 
                 frames_per_stream=2000, 
                 n_classes=2,
                 two_stage_training=False):
        
        assert os.path.exists(data_dir), f"{data_dir} does not exist."
        assert task in ["classification", "segmentation", "regression"], f"{task} is not valid."
        self.data_dir = data_dir
        
        self.file_list = np.array(os.listdir(data_dir))
        self.file_list = np.sort(self.file_list) # Sort file list to preserve temporal order
        self.file_list = self.file_list[selected_ids]
        self.file_list = np.sort(self.file_list)
        
        self.selected_ids = selected_ids # NOTE: 13 January 2025
        
        self.C = img_shape[0]
        self.H = img_shape[1]
        self.W = img_shape[2]
                                  
        self.task = task
        self.nb_files = len(self.file_list)
        self.FRAMES_PER_STREAM = frames_per_stream
        self.TOTAL_FRAMES = len(self.selected_ids) # Fixerd this for CloudL9_128
        
        self.n_classes = n_classes
        
        # Two-stage training
        self.cloud_threshold = cloud_threshold
        self.two_stage_training = two_stage_training
        if two_stage_training:
            self.first_stage_threshold = 0.3
            self.second_stage_threshold = cloud_threshold

    
    def __len__(self):
        return len(self.file_list)
    
    
    def __getitem__(self, index):
        if type(index) is int:
            index = np.array([index])
        
        X = torch.empty((len(index), self.C, self.H, self.W))
        
        if self.task in ["classification", "regression"] :
            Y = torch.empty(len(index))
        elif self.task == "segmentation":
            Y = torch.empty((len(index), self.H, self.W))
        start_time = time.time()
        for i, idx in enumerate(index):
            file_path = os.path.join(self.data_dir, self.file_list[idx])
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            stream = torch.from_numpy(data["stream_data"])
            cloud_mask = torch.from_numpy(data["cloudmask"])
            
            # For two-stage training
            cloudiness = torch.tensor(data["cloudiness"])
            
            # Normalization
            stream = stream / 255.0                
        
            if self.task in ["classification", "regression"]:
                X[i] = stream
                Y[i] = cloudiness
            elif self.task == "segmentation":
                X[i] =  stream
                Y[i] = cloud_mask
        if self.two_stage_training:
            return (X, Y)
        else:
            return (X, Y)
        
        
    def get_target(self, use_threshold=True):
            if self.task in ["classification", "regression"]:
                target = torch.empty(len(self.file_list))
                for i, file_name in enumerate(self.file_list):
                    file_path = os.path.join(self.data_dir, file_name)
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f) 
                    target[i] = data["cloudiness"].item()
                if use_threshold:
                    target = (target > self.cloud_threshold).to(torch.long)
                return target                
            elif self.task == "segmentation":
                raise NotImplementedError