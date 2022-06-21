import torch
from datasets.datasetVer1 import datasetVer1 as theDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from SAC_Agent import Agent
from util.getReward import getReward
from util.rewardDict import rewardDict

class MainLoop():

    def __init__(self,
                 baseDir,
                 windowSize,
                 gpuUse,
                 doEpiShuffle,
                 scalingMethod):
        super(MainLoop, self).__init__()

        self.baseDir = baseDir
        self.windowSize = windowSize
        self.scalingMethod = scalingMethod
        self.doEpiShuffle = doEpiShuffle
        self.gpuUse = gpuUse

        self.episodeDataset = theDataset(baseDir=self.baseDir,
                                         task='trn',
                                         windowSize=self.windowSize,
                                         scalingMethod=self.scalingMethod)

        self.episodeLoader = DataLoader(self.episodeDataset,
                                        batch_size=1,
                                        shuffle=self.doEpiShuffle,
                                        num_workers=1)

        self.Agent = Agent(input_dims=self.windowSize,
                           gpuUse=self.gpuUse)

        self.rewardDict = rewardDict

    def loadEpisodeOnMememory(self):

        stateLst = []
        labelLst = []

        for idx, (eachState, label) in enumerate(self.episodeLoader):

            stateLst.append(eachState)
            labelLst.append(label.item())

        return stateLst, labelLst

    def gymLikeStep(self):


    def doOneEpisode(self):

        for idx, (eachState,label) in enumerate(self.episodeLoader):

            if idx != 0:
                self.Agent.remember()

            done = False

            action = self.Agent.explore(state=eachState)

            reward = getReward(action,label.item(),rewardDict=rewardDict)









dir ='/home/a286winteriscoming/Downloads/TimeSeriesAnomalyDataset/Yahoo/' \
     'Yahoo/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'



loop = MainLoop(baseDir=dir,
                windowSize=17,
                gpuUse=True,
                doEpiShuffle=True,
                scalingMethod='minMax')

do = loop.doOneEpisode()
