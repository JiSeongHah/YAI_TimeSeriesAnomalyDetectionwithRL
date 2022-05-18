import torch
import pandas as pd 
import numpy as np

from torch.utils.data import Dataset, DataLoader

import os
import glob

# from config import get_parse


class YahooDataset(Dataset):

    def __init__(self, args):
        
        files_a1 = glob.glob(os.path.join(args.data_path,"A1Benchmark/real_*.csv"))  #list
        files_a1.sort()
        files_a2 = glob.glob(os.path.join(args.data_path,"A2Benchmark/synthetic_*.csv"))
        files_a2.sort()
        files_a3 = glob.glob(os.path.join(args.data_path,"A3Benchmark/A3Benchmark-TS*.csv"))
        files_a3.sort()
        files_a4 = glob.glob(os.path.join(args.data_path,"A4Benchmark/A4Benchmark-TS*.csv"))
        files_a4.sort()

        self.dataset = []
        for fn in files_a1:
            df = pd.read_csv(fn)
            self.dataset.append({
            'timestamp': (1483225200 + 3600 * df['timestamp']).tolist(),
            'value': df['value'].tolist(),
            'label': df['is_anomaly'].tolist()
            })
        for fn in files_a2:
            df = pd.read_csv(fn)
            self.dataset.append({
            'timestamp': df['timestamp'].tolist(),
            'value': df['value'].tolist(),
            'label': df['is_anomaly'].tolist()
            })
        for fn in files_a3:
            df = pd.read_csv(fn)
            self.dataset.append({
            'timestamp': df['timestamps'].astype(np.int_).tolist(),  
            'value': df['value'].tolist(),
            'label': df['anomaly'].tolist()
            })
        for fn in files_a4:
            df = pd.read_csv(fn)
            self.dataset.append({
            'timestamp': df['timestamps'].astype(np.int_).tolist(), 
            'value': df['value'].tolist(), 
            'label': df['anomaly'].tolist()
            })
        assert len(self.dataset) == 367, 'Incomplete dataset'        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        if torch.is_tnesor(idx):
            idx = idx.tolist()
        
        data = self.dataset[idx]
        time_stamps = np.array(data['timestamp'])
        values = np.array(data['value'])
        labels = np.array(data['label']) 

        l = int(len(labels) * 0.8) # Spliting data portion into train and test
        
        train_time_stamps = time_stamps[:l]
        train_datas = values[:l]  
        train_labels = labels[:l]     # 

        test_time_stmaps = time_stamps[l:]        
        test_datas = values[l:]
        test_labels = labels[l:]


        # normalize
        train_datas = (train_datas - train_datas.mean()) / train_datas.std()
        test_datas = (test_datas - test_datas.mean()) / test_datas.std()

        
        return (train_time_stamps, train_datas, train_labels), (test_time_stmaps,test_datas,test_labels)




# if __name__=="__main__":
#     args = get_parse()
#     Yahoo_data = YahooDataset()




