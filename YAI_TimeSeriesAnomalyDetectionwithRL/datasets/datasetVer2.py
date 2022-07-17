
import torch
import os
import pickle
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
import json
import random
import csv


class datasetVer2(Dataset):
    def __init__(self,
                 baseDir,
                 task,
                 windowSize,
                 scalingMethod):
        super(datasetVer2, self).__init__()

        self.baseDir = baseDir
        self.task = task
        self.windowSize = windowSize
        assert type(self.windowSize) == int
        self.scalingMethod = scalingMethod

        if self.task == 'trn':
            # assume that there's already trainDir folder by splitData module
            self.dataFolderDir = self.baseDir+'trainDir/'
        else:
            # assume that there's already testDir folder by splitData module
            self.dataFolderDir = self.baseDir + 'testDir/'

        # select certain csv file to load
        self.pickedFolder = random.choice(os.listdir(self.dataFolderDir))


        # open csv file and change into torch tensor
        with open(self.dataFolderDir+self.pickedFolder,'r') as f:
            rdr = csv.reader(f)
            self.dataArr = np.asarray(list(rdr)[1:]).astype(float)

        self.dataArr = torch.from_numpy(self.dataArr)

        self.inputArr = self.dataArr[:,1]
        self.labelArr = self.dataArr[:,2]

        if self.scalingMethod == 'minMax':
            MaxValue = torch.max(self.inputArr)
            MinValue = torch.min(self.inputArr)

            self.inputArr = (self.inputArr -MinValue)/ (MaxValue- MinValue)

        elif self.scalingMethod == 'zScore':
            MeanValue = torch.mean(self.inputArr)
            stdValue = torch.std(self.inputArr)

            self.inputArr = (self.inputArr-MeanValue)/stdValue


    def __len__(self):

        return len(self.dataArr) -1*self.windowSize -1


    def __getitem__(self, idx):

        # data slice
        inputData = self.inputArr[idx:idx+self.windowSize]
        labels = self.labelArr[idx:idx+self.windowSize]

        # if len(labels) != 17:
        #     print(f'size is :{labels.size()}, folder is : {self.pickedFolder}, idx : {idx}')
        # print(f'labels is : {labels.size()} with len : {len(labels)} with folder : {self.pickedFolder} and idx :{idx}')
        # because input data is on column 1
        # print(f'staet bef is : {inputData} with dir : {self.pickedFolder}')

        #choose label of last time stamp
        # because label is on the column 2

        label = labels[-1]

        # print(f'label is : {label} and size : {label.size()}')

        return inputData, label

    # change csv file to use different file
    def changeFolder(self):

        exFolder = self.pickedFolder
        self.pickedFolder = random.choice(os.listdir(self.dataFolderDir))
        print(f'changing from {exFolder} to current : {self.pickedFolder} complete')

        with open(self.dataFolderDir+self.pickedFolder,'r') as f:
            rdr = csv.reader(f)
            self.dataArr = np.asarray(list(rdr)[1:]).astype(float)
        self.dataArr = torch.from_numpy(self.dataArr)

        self.inputArr = self.dataArr[:,1]
        self.labelArr = self.dataArr[:,2]
        print(f'lnef of dataArr is : {len(self.dataArr)} while len of inputArr is : {len(self.inputArr),len(self.labelArr)}')





# dir ='/home/a286winteriscoming/Downloads/TimeSeriesAnomalyDataset/Yahoo/' \
#      'Yahoo/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'
#
#
# DATASET = datasetVer2(baseDir=dir,
#                       task='trn',
#                       windowSize=13,
#                       scalingMethod='minMax')
#
#
# dl = DataLoader(DATASET,batch_size=1,shuffle=True)
# for inputs,label in dl:
#     print(inputs.size(),label)

# for Input,label in DATASET:
#
#     print(Input.size(),label)


