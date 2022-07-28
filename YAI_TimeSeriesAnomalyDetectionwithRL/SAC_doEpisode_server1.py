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


class MainLoop():

    def __init__(self,
                 baseDir,
                 resultSaveDir,
                 windowSize,
                 beta,
                 gpuUse,
                 doEpiShuffle,
                 batchSize,
                 updateTargetNetTerm,
                 scalingMethod):
        super(MainLoop, self).__init__()

        self.baseDir = baseDir
        self.resultSaveDir = resultSaveDir
        self.windowSize = windowSize
        self.beta = beta
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

        self.validationDataset = theDataset(baseDir=self.baseDir,
                                         task='tst',
                                         windowSize=self.windowSize[0],
                                         scalingMethod=self.scalingMethod)

        self.validationLoader = iter(DataLoader(self.validationDataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1))

        self.Agent = Agent(input_dims=self.windowSize,
                           batch_size=self.batchSize,
                           plotSaveDir=self.baseDir+'dir1/',
                           gpuUse=self.gpuUse)

        self.rewardDict = rewardDict

        self.TPLstTrn = []
        self.FPLstTrn = []
        self.FNLstTrn = []
        self.TNLstTrn = []

        self.precisionTrn = []
        self.RecallTrn = []
        self.f1BetaScoreTrn = []

        self.TPLstVal = []
        self.FPLstVal = []
        self.FNLstVal = []
        self.TNLstVal = []

        self.precisionVal = []
        self.RecallVal = []
        self.f1BetaScoreVal = []

    def flushLst(self):
        self.TPLstTrn.clear()
        self.FPLstTrn.clear()
        self.FNLstTrn.clear()
        self.TNLstTrn.clear()

        self.TPLstVal.clear()
        self.FPLstVal.clear()
        self.FNLstVal.clear()
        self.TNLstVal.clear()

        print(' flushing mainloop complete ')


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

    def gymLikeResetVal(self):

        # load new data and start new episode
        self.validationDataset.changeFolder()
        self.validationLoader = iter(DataLoader(self.validationDataset,
                                             batch_size=1,
                                             shuffle=self.doEpiShuffle,
                                             num_workers=1))

        firstState, firstLabel = next(self.validationLoader)

        return firstState, firstLabel

    def gymLikeStepVal(self,action,label,idx):

        nextState, nextLabel = next(self.validationLoader)

        reward = getReward(action,label.item(),rewardDict=rewardDict)

        if idx == len(self.validationLoader) -1:
            done = True
        else:
            done = False

        return nextState,nextLabel,reward,done


    def trnOneEpisode(self):

        state,label = self.gymLikeReset()

        # print(f'len of episodeloader is : {len(self.episodeLoader)}')
        # print(f'state is : {state}')

        if torch.sum(torch.isnan(state.clone().detach()).float()):
            print(f'state is nan for dir : {self.episodeDataset.pickedFolder} with idx : 0')


        for idx in range(len(self.episodeLoader)-1):

            action = self.Agent.explore(state)
            nextState, nextLabel, reward, done = self.gymLikeStep(action=action,label=label,idx=idx)

            if torch.sum(torch.isnan(nextState.clone().detach()).float()):
                print(f'state is nan for dir : {self.episodeDataset.pickedFolder} with idx : {idx}'
                      f'and  states is {nextState}')

            if reward[1] == 'TP':
                self.TPLstTrn.append(1)
            elif reward[1] == 'FP':
                self.FPLstTrn.append(1)
            elif reward[1] == 'FN':
                self.FNLstTrn.append(1)
            else:
                self.TNLstTrn.append(1)

            # print(f'state is :{state.size()} with path : {self.episodeDataset.pickedFolder} wit idx :{idx}')
            # print(f'nextState is :{nextState.size()}')
            self.Agent.remember(state=state,
                                action=action,
                                reward=reward[0],
                                new_state=nextState,
                                done=done)

            state,label = nextState, nextLabel

            if self.Agent.memory.mem_cntr > self.batchSize:
                self.Agent.learn()

            if idx % self.updateTargetNetTerm == 0 and self.Agent.memory.mem_cntr > self.batchSize:
                self.Agent.updateTarget()
                self.Agent.plotAvgLosses()


    def validateOneEpisode(self):
        print('start validating ...')
        print('start validating ...')
        print('start validating ...')
        print('start validating ...')

        state, label = self.gymLikeResetVal()

        for idx in range(len(self.validationLoader) - 1):

            action = self.Agent.exploit(state)
            nextState, nextLabel, reward, done = self.gymLikeStepVal(action=action, label=label, idx=idx)

            label = nextLabel

            if reward[1] == 'TP':
                self.TPLstVal.append(1)
            elif reward[1] == 'FP':
                self.FPLstVal.append(1)
            elif reward[1] == 'FN':
                self.FNLstVal.append(1)
            else:
                self.TNLstVal.append(1)


    def validationStepEnd(self):

        ######### Calculate f1 beta score of training one episode ###################################

        # print(1,np.sum(self.TPLstTrn))
        # print(2,np.sum(self.FPLstTrn))
        # print(3,np.sum(self.FNLstTrn))
        # print(33,np.sum(self.TNLstTrn))

        if (np.sum(self.TPLstTrn) + np.sum(self.FPLstTrn) ) ==0:
            self.precisionTrn.append(0)
        else:
            self.precisionTrn.append(np.sum(self.TPLstTrn) / (np.sum(self.TPLstTrn) + np.sum(self.FPLstTrn) + 1e-9))

        if (np.sum(self.TPLstTrn) + np.sum(self.FNLstTrn) ) == 0:
            self.RecallTrn.append(0)
        else:
            self.RecallTrn.append(np.sum(self.TPLstTrn) / (np.sum(self.TPLstTrn) + np.sum(self.FNLstTrn) + 1e-9))

        # print(self.precisionTrn[-1],self.RecallTrn[-1])

        if (self.precisionTrn[-1] * self.RecallTrn[-1] == 0):
            latestF1BetaScore = 0
        else:
            latestF1BetaScore = ((1 + self.beta ** 2) * self.precisionTrn[-1] * self.RecallTrn[-1]) / \
                                ((self.beta ** 2) * self.precisionTrn[-1] + self.RecallTrn[-1])
        self.f1BetaScoreTrn.append(latestF1BetaScore)
        ######### Calculate f1 beta score of training one episode ###################################


        ######### Calculate f1 beta score of validation one episode ###################################
        # print(77,len(self.validationLoader))
        # print(4,np.sum(self.TPLstVal))
        # print(5,np.sum(self.FPLstVal))
        # print(6,np.sum(self.FNLstVal))
        # print(7,np.sum(self.TNLstVal))

        if ( np.sum(self.TPLstVal) + np.sum(self.FPLstVal) ) == 0:
            self.precisionVal.append(0)
        else:
            self.precisionVal.append(np.sum(self.TPLstVal) / (np.sum(self.TPLstVal) + np.sum(self.FPLstVal) + 1e-9))

        if ( np.sum(self.TPLstVal) + np.sum(self.FNLstVal)) == 0:
            self.RecallVal.append(0)
        else:
            self.RecallVal.append(np.sum(self.TPLstVal) / (np.sum(self.TPLstVal) + np.sum(self.FNLstVal) + 1e-9))

        # print(self.precisionVal[-1], self.RecallVal[-1])

        if (self.precisionVal[-1] * self.RecallVal[-1]) == 0:
            latestF1BetaScore = 0
        else:
            latestF1BetaScore = ((1 + self.beta ** 2) * self.precisionVal[-1] * self.RecallVal[-1]) / \
                                ((self.beta ** 2) * self.precisionVal[-1] + self.RecallVal[-1])
        self.f1BetaScoreVal.append(latestF1BetaScore)
        ######### Calculate f1 beta score of validation one episode ###################################

        fig = plt.figure(constrained_layout= True)
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(range(len(self.precisionTrn)), self.precisionTrn)
        ax1.set_xlabel('episode')
        ax1.set_title('Precision Trn')

        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(range(len(self.RecallTrn)), self.RecallTrn)
        ax2.set_xlabel('episode')
        ax2.set_title('Recall Trn')

        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(range(len(self.f1BetaScoreTrn)), self.f1BetaScoreTrn)
        ax3.set_xlabel('episode')
        ax3.set_title('f1 beta score Trn')

        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(range(len(self.precisionVal)), self.precisionVal)
        ax4.set_xlabel('episode')
        ax4.set_title('Precision Val')

        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(range(len(self.RecallVal)), self.RecallVal)
        ax5.set_xlabel('episode')
        ax5.set_title('Recall Val')

        ax6 = fig.add_subplot(3, 2, 6)
        ax6.plot(range(len(self.f1BetaScoreVal)), self.f1BetaScoreVal)
        ax6.set_xlabel('episode')
        ax6.set_title('f1 beta score Val')

        plt.savefig(self.resultSaveDir+'scoreResult.png',dpi=200)
        print(f'episode complete with TP : {np.sum(self.TPLstVal)}, FN : {np.sum(self.FNLstVal)}'
              f', FP : {np.sum(self.FPLstVal)} , TN : {np.sum(self.TNLstVal)}')
        plt.close()
        plt.cla()
        plt.clf()

        print(f'episode complete with f1 score : {latestF1BetaScore} ,'
              f' from precision : {self.precisionVal[-1]} and recall : {self.RecallVal[-1]} for beta :{self.beta}')

        with open(self.resultSaveDir+'scoreLst.pkl','wb') as F:
            pickle.dump(self.f1BetaScoreVal,F)

        self.flushLst()



    def StartTrnAndVal(self,iterNum):

        for iteration in range(iterNum):

            self.trnOneEpisode()
            self.validateOneEpisode()
            self.validationStepEnd()



os.environ['CUDA_VISIBLE_DEVICES'] = "1"

dir ='/home/a286/hjs_dir1/myYAI_RL0/TimeSeriesAnomalyDataset/Yahoo/' \
     'Yahoo/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'

scalingMethodLst = ['minMax','zScore']
doShuffle = [True,False]

scalingMethod = scalingMethodLst[0]
SHUFFLE = doShuffle[1]


wSizeLst = [10*i for i in range(1,30)]
updateTargetTermLst = [2**(i+1) for i in range(4,10)]


for updateTargetTerm in updateTargetTermLst:
    for wSize in wSizeLst:

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
                        doEpiShuffle=SHUFFLE,
                        updateTargetNetTerm=updateTargetTerm,
                        scalingMethod=scalingMethod)

        do = loop.StartTrnAndVal(3000)













































































