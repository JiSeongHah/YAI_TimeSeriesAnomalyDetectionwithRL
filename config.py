import argparse
import torch

# Params
def get_parse():
    args = argparse.Namespace()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #data
    args.datasets = 'Yahoo'  # Yahoo, SWaT, Numenta, KPI
    
    args.data_path = 'dataset/'
    args.batch_size = 53 
    args.shuffle = False
    args.window_size = 50   
    args.split_ratio = 0.8

    #reward
    args.TP = 5
    args.TN = 1
    args.FP = -10
    args.FN = -3

    #parameters
    args.replay_batch_size = 30
    args.buffersize = 1000


    #ICM
    args.lambda_ = 0.1 #policy loss
    args.beta_ = 0.2  #inverse, forward loss

    # eps greedy policy
    args.eps = 0.1

    #train
    args.epochs = 300
    args.gamma = 0.2 # for Q networ train


    return args    

