import numpy as np
import os
from usefulFuncs import createDirectory
import csv
import shutil
import random



def splitData(dataDir,splitRatio,randomPick=True):

    fileLst = [i for i in os.listdir(dataDir) if i.endswith('.csv')]
    fullDirLst = sorted(fileLst,key=lambda x:int(x.split('_')[-1].split('.')[0]))

    if randomPick == True:
        random.shuffle(fullDirLst)
        random.shuffle(fullDirLst)
        random.shuffle(fullDirLst)

    splitNum = int(len(fullDirLst) * splitRatio)

    trainLst = fullDirLst[:splitNum]
    testLst = fullDirLst[splitNum:]
    createDirectory(dataDir+'trainDir')
    createDirectory(dataDir+'testDir')

    for eachTrainFile in trainLst:
        shutil.copy(dataDir+eachTrainFile,dataDir+'trainDir/'+eachTrainFile)
        print(f'copying {eachTrainFile} complete')

    for eachTestFile in testLst:
        shutil.copy(dataDir+eachTestFile,dataDir+'testDir/'+eachTestFile)
        print(f'copying {eachTestFile} complete')


if __name__ == '__main__':

    dataDir = '/home/emeraldsword1423/Downloads/Yahoo/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'

    splitData(dataDir=dataDir,splitRatio=0.8)