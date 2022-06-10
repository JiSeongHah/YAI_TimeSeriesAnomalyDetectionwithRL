import torch
import os
import pickle
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
import json
import random
import csv


class datasetVer1(Dataset):
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


    def __len__(self):

        return len(self.dataArr) -self.windowSize + 1


    def __getitem__(self, idx):

        # data slice
        dataInScope = self.dataArr[idx:idx+self.windowSize,:]
        # because input data is on column 1
        inputData = dataInScope[:,1]

        if self.scalingMethod == 'minMax':
            MaxValue = torch.max(inputData)
            inputData = inputData / MaxValue
        elif self.scalingMethod == 'zScore':
            MeanValue = torch.mean(inputData)
            stdValue = torch.std(inputData)

            inputData = (inputData-MeanValue)/stdValue

        #choose label of last time stamp
        # because label is on the column 2
        label = dataInScope[-1,2]


        return inputData,label

    # change csv file to use different file
    def changeFolder(self):

        exFolder = self.pickedFolder
        self.pickedFolder = random.choice(os.listdir(self.dataFolderDir))
        print(f'changing from {exFolder} to current : {self.pickedFolder} complete')

        with open(self.dataFolderDir+self.pickedFolder,'r') as f:
            rdr = csv.reader(f)
            self.dataArr = np.asarray(list(rdr)[1:]).astype(float)

        self.dataArr = torch.from_numpy(self.dataArr)