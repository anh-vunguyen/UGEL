"""Stream Dataset
Authors: Anh Vu Nguyen
"""
import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import time
import logging
import gc 

class DatasetOTF(Dataset):
    def __init__(self, path, name=None, selected_ids=None, nb_frames_per_file=5000):
        assert os.path.exists(path), f"{path} does not exist."
        self.name = name
        self.path = path
        self.file_list = np.array(os.listdir(path))
        self.file_list  = np.sort(self.file_list)
        if selected_ids is not None:
            self.file_list = self.file_list[selected_ids]
            self.selected_ids = selected_ids # NOTE: 13 January  2025
        self.nb_frames_per_file = nb_frames_per_file
        self.current_stream_id = -1
    
    
    def __len__(self):
        return self.nb_frames_per_file * len(self.file_list)
    
    
    def __getitem__(self, index):
        if type(index) is not list:
            index = [index]
        X = torch.empty((len(index), self.C, self.H, self.W))
        Y = torch.empty(len(index), dtype=torch.long)
        for i, idx in enumerate(index):
            if (idx // self.nb_frames_per_file) != self.current_stream_id:
                file_path = os.path.join(self.path, self.file_list[idx // self.nb_frames_per_file])
                start_time = time.time()
                with open(file_path, 'rb') as f:
                    data_dict = pickle.load(f)
                print(f"Loading time for {file_path}: {time.time()-start_time} s.")    
                data = data_dict["data"]
                label = data_dict["label"]
                data = data / 255.0  # Normalization             
                self.current_stream_id = idx // self.nb_frames_per_file
            X[i] = data[idx % self.nb_frames_per_file].unsqueeze(0)
            Y[i] = label[idx % self.nb_frames_per_file].unsqueeze(0)
        return (X, Y)
    
    
    def get_target(self):
        target = torch.empty(self.nb_frames_per_file * len(self.file_list))
        for i, file_name in enumerate(self.file_list):
            file_path = os.path.join(self.path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f) 
            target[i*self.nb_frames_per_file:(i+1)*self.nb_frames_per_file] = data["label"]
            del data
            gc.collect()
        return target                


class DataHandlerOTF(Dataset):
    def __init__(self, sub_dataset, transform=None, threshold=0.7):
        self.X, self.Y = sub_dataset # Load on-the-fly
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index
    
    def __len__(self):
        return len(self.Y)