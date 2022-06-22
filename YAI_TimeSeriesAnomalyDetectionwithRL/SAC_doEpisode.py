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
                 batchSize,
                 updateTargetNetTerm,
                 scalingMethod):
        super(MainLoop, self).__init__()

        self.baseDir = baseDir
        self.windowSize = windowSize
        self.scalingMethod = scalingMethod
        self.doEpiShuffle = doEpiShuffle
        self.batchSize = batchSize
        self.updateTargetNetTerm = updateTargetNetTerm
        self.gpuUse = gpuUse


        self.episodeDataset = theDataset(baseDir=self.baseDir,
                                         task='trn',
                                         windowSize=self.windowSize[0],
                                         scalingMethod=self.scalingMethod)

        self.episodeLoader = iter(DataLoader(self.episodeDataset,
                                        batch_size=1,
                                        shuffle=self.doEpiShuffle,
                                        num_workers=1))

        self.Agent = Agent(input_dims=self.windowSize,
                           batch_size=self.batchSize,
                           plotSaveDir=self.baseDir+'dir1/',
                           gpuUse=self.gpuUse)

        self.rewardDict = rewardDict


    def loadEpisodeOnMememory(self):

        stateLst = []
        labelLst = []

        for idx, (eachState, label) in enumerate(self.episodeLoader):

            stateLst.append(eachState)
            labelLst.append(label.item())

        return stateLst, labelLst

    def gymLikeReset(self):

        # load new data and start new episode
        self.episodeDataset.changeFolder()
        self.episodeLoader = iter(DataLoader(self.episodeDataset,
                                             batch_size=1,
                                             shuffle=self.doEpiShuffle,
                                             num_workers=1))

        firstState, firstLabel = next(self.episodeLoader)

        return firstState, firstLabel

    def gymLikeStep(self,action,label,idx):

        nextState, nextLabel = next(self.episodeLoader)

        reward = getReward(action,label.item(),rewardDict=rewardDict)

        if idx == len(self.episodeLoader) -2:
            done = True
        else:
            done = False

        return nextState,nextLabel,reward,done


    def doOneEpisode(self):

        state,label = self.gymLikeReset()

        for idx in range(len(self.episodeLoader)-1):


            action = self.Agent.explore(state)
            nextState, nextLabel, reward, done = self.gymLikeStep(action=action,label=label,idx=idx)

            self.Agent.remember(state=state,
                                action=action,
                                reward=reward,
                                new_state=nextState,
                                done=done)

            state,label = nextState, nextLabel

            if self.Agent.memory.mem_cntr > self.batchSize:
                self.Agent.learn()

            if idx % self.updateTargetNetTerm == 0 and self.Agent.memory.mem_cntr > self.batchSize:
                self.Agent.updateTarget()
                self.Agent.plotAvgLosses()









dir ='/home/a286winteriscoming/Downloads/TimeSeriesAnomalyDataset/Yahoo/' \
     'Yahoo/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'



loop = MainLoop(baseDir=dir,
                windowSize=[17],
                batchSize=256,
                gpuUse=True,
                doEpiShuffle=True,
                updateTargetNetTerm=10,
                scalingMethod='minMax')

do = loop.doOneEpisode()


lst = []

for idx ,i in enumerate(range(1,5)):

    if idx !=0:
        lst.append([state,done,action,reward,])

    state = str(i)

    done = False

    action = i

    reward = -i*i











































































