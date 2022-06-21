import numpy as np

def getReward(action,label,rewardDict):

    assert action in [0,1], 'action must be 0 or 1'
    assert label in [0,1], 'label must be 0 or 1'

    #case 0 : True Positive
    if action == label and label== 1:
        return rewardDict['TP']

    elif action != label and label == 1:
        return rewardDict['FN']

    elif action != label and label == 0:
        return rewardDict['FP']

    else:
        return rewardDict['TN']

#
# rewardDict= {
#     'TP': 1,
#     'FN':2,
#     'FP':3,
#     'TN':4
# }
#
# aLst = [0,1]
# bLst = [0,1]
#
# for a in aLst:
#     for b in bLst:
#         print(getReward(a,b,rewardDict))
