from collections import defaultdict
from email.policy import default
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np 
import pickle
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from utils.config_predator_prey import Config

class OpponentEDataset(Dataset):
    def __init__(self, data_file, train=True):
        if train:
            self.N = 10000
        else:
            self.N = 1000
        self.data_i, self.data_j, self.data_plus, self.data_r, self.data_minus = self.read_pickle(data_file)
        
    def __getitem__(self, index):
        return self.data_i[index], self.data_j[index], self.data_plus[index], self.data_r[index], self.data_minus[index]

    def __len__(self):
        return len(self.data_i)

    def read_pickle(self, data_file):
        data = pd.read_pickle(data_file)
        tau_dict = data['data_tau']
        for key in tau_dict:
            tau_dict[key] = np.array(tau_dict[key])
            # print(tau_dict[key].shape)
        
        data_i = []
        data_j = []
        data_plus = []
        data_r = []
        data_minus = []
                
        for _ in range(self.N):
            for i in tau_dict.keys():
                for j in tau_dict.keys():
                    if i != j:
                        index1 = np.random.choice(range(0, len(tau_dict[i])), size=2, replace=False)
                        index2 = np.random.choice(range(0, len(tau_dict[j])), size=1)
                        data_i.append(i)
                        data_j.append(j)
                        data_plus.append(tau_dict[i][index1[0]])
                        data_r.append(tau_dict[i][index1[1]])
                        data_minus.append(tau_dict[j][index2[0]])

        data_i = np.array(data_i)
        data_j = np.array(data_j)
        data_plus = np.array(data_plus, dtype=np.float32)
        data_r = np.array(data_r, dtype=np.float32)
        data_minus = np.array(data_minus, dtype=np.float32)
        
        print(data_i.shape, data_j.shape, data_plus.shape, data_r.shape, data_minus.shape)
        
        return data_i, data_j, data_plus, data_r, data_minus


if __name__ == "__main__":
    data_dir = Config.DATA_DIR
    version = "v4"
    test_data_file = data_dir + 'e_data_simple_tag_' + version + '.p'
    test_dset = OpponentEDataset(test_data_file)