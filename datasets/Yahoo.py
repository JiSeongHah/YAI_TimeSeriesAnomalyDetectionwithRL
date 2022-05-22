import torch
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob
from config import get_parse
from util import sliding_window 

def build_yahoo(args):
    train, test = Yahoo_Dataprocessing(args)

    train_state = sliding_window(train, args)
    test_state = sliding_window(test, args)

    train_set = YahooDataset(train_state)
    test_set = YahooDataset(test_state)

    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = 2)
    test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = 2) 

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
        if torch.is_tnesor(idx):
            idx = idx.tolist()
        
        timestamp = self.timestamp[idx]
        value = self.value[idx]
        label = self.label[idx]

        time_stamp = np.array(timestamp)
        value = np.array(value) 
        label = np.array(label) 

        # normalize
        value = (value - value.mean()) / value.std()

        return time_stamp, value, label

def Yahoo_Dataprocessing(args):
    
    files_a1 = glob.glob(os.path.join(args.data_path,"A1Benchmark/real_*.csv"))  #list
    files_a1.sort()
    files_a2 = glob.glob(os.path.join(args.data_path,"A2Benchmark/synthetic_*.csv"))
    files_a2.sort()
    files_a3 = glob.glob(os.path.join(args.data_path,"A3Benchmark/A3Benchmark-TS*.csv"))
    files_a3.sort()
    files_a4 = glob.glob(os.path.join(args.data_path,"A4Benchmark/A4Benchmark-TS*.csv"))
    files_a4.sort()

    dataset = []    
    for fn in files_a1:
        df = pd.read_csv(fn)
        dataset.append({
            'timestamp': (1483225200 + 3600 * df['timestamp']).tolist(),
            'value': df['value'].tolist(),
            'label': df['is_anomaly'].tolist()
            })
    for fn in files_a2:
            df = pd.read_csv(fn)
            dataset.append({
            'timestamp': df['timestamp'].tolist(),
            'value': df['value'].tolist(),
            'label': df['is_anomaly'].tolist()
            })
    for fn in files_a3:
            df = pd.read_csv(fn)
            dataset.append({
            'timestamp': df['timestamps'].astype(np.int_).tolist(),  
            'value': df['value'].tolist(),
            'label': df['anomaly'].tolist()
            })
    for fn in files_a4:
            df = pd.read_csv(fn)
            dataset.append({
            'timestamp': df['timestamps'].astype(np.int_).tolist(), 
            'value': df['value'].tolist(), 
            'label': df['anomaly'].tolist()
            })
    assert len(dataset) == 367, 'Error'

    split_bar = 255
        
    train = dataset[:split_bar] # list of sets
    test = dataset[split_bar:]

    return train, test
