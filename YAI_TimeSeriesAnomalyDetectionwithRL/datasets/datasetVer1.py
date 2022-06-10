import torch
import os
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json
import random
import csv


class datasetVer1(torch.utils.data.Dataset):
    
    def __init__(self,
                 baseDir,
                 task,
                 windowSize,
                 scalingMethod):
        super(datasetVer1, self).__init__()

        self.baseDir = baseDir
        self.task = task
        self.windowSize = windowSize
        assert type(self.windowSize) == int
        self.scalingMethod = scalingMethod

        if self.task == 'trn':
            dataFolderDir = self.baseDir+'trainDir/'
        else:
            dataFolderDir = self.baseDir + 'testDir/'

        self.pickedFolder = random.choice(os.listdir(dataFolderDir))

        with open(dataFolderDir+self.pickedFolder,'r') as f:
            rdr = csv.reader(f)
            self.DATAArr = np.asarray(list(rdr)[1:]).astype(float)

        self.DATAArr = torch.from_numpy(self.DATAArr)

        self.dataArr = torch.zeros(30,5)


    def __len__(self):

        return int(len(self.dataArr)/100)

        # return len(self.dataArr) - self.windowSize


    def __getitem__(self, idx):

        dataInScope = self.DATAArr[idx:idx+self.windowSize,:]

        inputData = dataInScope[:,1]

        if self.scalingMethod == 'minMax':
            MaxValue = torch.max(inputData)
            inputData = inputData / MaxValue
        elif self.scalingMethod == 'zScore':
            MeanValue = torch.mean(inputData)
            stdValue = torch.std(inputData)

            inputData = (inputData-MeanValue)/stdValue

        label = dataInScope[-1,2]
        print('idx is : ',idx)
        print('total len is : ',len(self.dataArr))

        return inputData,label



dir ='/home/a286winteriscoming/Downloads/TimeSeriesAnomalyDataset/Yahoo/' \
     'Yahoo/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'


DATASET = datasetVer1(baseDir=dir,
                      task='trn',
                      windowSize=13,
                      scalingMethod='minMax')

for Input,label in DATASET:

    print(Input.size(),label)


