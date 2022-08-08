import csv
import os
import torch

import numpy as np
#
# dir ='/home/a286winteriscoming/Downloads/TimeSeriesAnomalyDataset/Yahoo/' \
#      'Yahoo/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/trainDir/'
#
# with open(dir+'real_25.csv','r') as f:
#     rdr =csv.reader(f)
#     lst = np.asarray(list(rdr)[1:])[:,1].astype(float)
#
# for i in lst:
#     print(i)
#
x = torch.randn(3,128)
y= torch.randn(8,128)
z= torch.cat((x,y))
print(z.size())
print(z.size(0))