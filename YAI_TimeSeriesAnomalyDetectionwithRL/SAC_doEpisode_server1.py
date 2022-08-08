import torch
from datasets.datasetVer2 import datasetVer2 as theDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from SAC_Agent import Agent
from util.getReward import getReward
from util.rewardDict import rewardDict
from util.usefulFuncs import mk_name,createDirectory
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from SAC_mainLoop import MainLoop



os.environ['CUDA_VISIBLE_DEVICES'] = "1"

dir ='/home/a286/hjs_dir1/myYAI_RL0/TimeSeriesAnomalyDataset/Yahoo/' \
     'Yahoo/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'

scalingMethodLst = ['minMax','zScore']
doShuffle = [True,False]

scalingMethod = scalingMethodLst[0]
SHUFFLE = doShuffle[0]


anomalyRatio = 0.2

wSize= 25
updateTargetTerm = 32

resultSaveDir = dir + mk_name(shuffle=SHUFFLE,
                              wSize=wSize,
                              scalingMethod=scalingMethod,
                              updateTerm=updateTargetTerm) + '/'
createDirectory(resultSaveDir)

loop = MainLoop(baseDir=dir,
                resultSaveDir=resultSaveDir,
                windowSize=[wSize],
                batchSize=256,
                beta=1,
                gpuUse=True,
                anomalyRatio=anomalyRatio,
                doEpiShuffle=SHUFFLE,
                updateTargetNetTerm=updateTargetTerm,
                scalingMethod=scalingMethod)

do = loop.StartTrnAndVal(501)
print('testline2')
















































































