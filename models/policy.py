from gettext import npgettext
import numpy as np
import torch
import torch.nn.functional as F



def epsilon_greedy_policy(qvals,eps):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=0,high=1,size=(1,)) # 0 : normal, 1 : anomaly
        else:
            return torch.argmax(qvals)
    else:
        return torch.multinomial(F.softmax(F.normalize(qvals)), num_samples=1)

#TODO
'''if u want to use different policy'''