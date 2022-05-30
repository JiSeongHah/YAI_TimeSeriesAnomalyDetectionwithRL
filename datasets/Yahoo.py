from audioop import minmax
import torch
import pandas as pd 
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import minmax_scale

import os
import glob

from util.sliding_window import sliding_window 



def build_yahoo(args): 
    train, test = Yahoo_Dataprocessing(args) # divide data into train, test : list of sets

    train = sliding_window(train, args)  # sliding window
    test = sliding_window(test, args)

    train = YahooDataset(train)
    test = YahooDataset(test)

    train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = 2)
    test_loader = DataLoader(test, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = 2) 

    return train_loader, test_loader

class YahooDataset(Dataset):

    def __init__(self, dataset):
        super(YahooDataset,self).__init__()

        self.timestamp = dataset['timestamp']
        self.value = dataset['value']
        self.label = dataset['label']

    def __len__(self):
        return len(self.timestamp)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        timestamp = self.timestamp[idx]
        value = self.value[idx]
        label = self.label[idx][-1]

        time_stamp = np.array(timestamp)
        value = np.array(value) 
        label = np.array(label)

        return time_stamp, value, label


def Yahoo_Dataprocessing(args):
    
    files_a1 = glob.glob(os.path.join(args.data_path,"A1Benchmark/real_*.csv"))  #list
    files_a1.sort()
    files_a2 = glob.glob(os.path.join(args.data_path,"A2Benchmark/synthetic_*.csv"))
    files_a2.sort()

    split_bar = np.array([len(files_a1),len(files_a2)])
    split_bar = args.split_ratio * split_bar
    split_bar = np.asarray(split_bar,dtype = int)
    
    train = []    
    test = []
    for i, fn  in enumerate(files_a1):
        df = pd.read_csv(fn)
        if i < split_bar[0]: # train
            train.append({
            'timestamp': (df['timestamp']).tolist(),
            'value': minmax_scale(df['value'].tolist()),
            'label': df['is_anomaly'].tolist()
            })
        else: #test
            test.append({
            'timestamp': (df['timestamp']).tolist(),
            'value': minmax_scale(df['value'].tolist()),
            'label': df['is_anomaly'].tolist()
            })
    for i, fn in enumerate(files_a2):
        df = pd.read_csv(fn)
        if i < split_bar[1]: # train
            train.append({
                'timestamp': df['timestamp'].tolist(),
                'value': minmax_scale(df['value'].tolist()),
                'label': df['is_anomaly'].tolist()
                })
        else: #tesst
            test.append({
            'timestamp': (df['timestamp']).tolist(),
            'value': minmax_scale(df['value'].tolist()),
            'label': df['is_anomaly'].tolist()
            })
    assert len(train) + len(test)  == 167, 'Error'

    return train, test
