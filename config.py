import argparse
from platform import java_ver
import torch

# Params
def get_parse():
    args = argparse.Namespace()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #data
    args.datasets = 'Yahoo'  # Yahoo, SWaT, Numenta, KPI
    args.using_data = 'A1' #'A1', 'A2'
    args.data_path = 'G:\내 드라이브\YAI\AD_for_TS_w_RL\data\Yahoo\ydata-labeled-time-series-anomalies-v1_0'

    args.shuffle = False
    args.window_size = 50   
    args.split_ratio = 0.8


    #parameters
    args.batch_size = 150

    print('data set : ', args.using_data)

    return args    