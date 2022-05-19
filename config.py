import argparse
import torch

# Params
def get_parse():
    args = argparse.Namesapce()

    #TODO
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #data
    args.datasets = 'Yahoo' # Yahoo, SWaT, Numenta, KPI 
    args.window_size = #
    args.input_size = #

    #parameters
    args.batch_size = 150
    args.gamma = 0.2
    args.max_episode_len = 100
    

    #model
    args.model = 'ICM'

    #train
    '''
    forward_loss = nn.MSELoss(reduction='none')
    inverse_loss = nn.CrossEntropyLoss(reduction='none')
    qloss = nn.MSELoss()
    '''
    args.optimizer = 'Adam'
    args.epochs = 300    




    #policy
    args.eps = 0.2

    

