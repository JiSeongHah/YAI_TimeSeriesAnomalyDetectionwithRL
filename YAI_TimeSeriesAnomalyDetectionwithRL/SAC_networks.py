import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from torch.distributions import Categorical

class QNetwork(nn.Module):
    def __init__(self,
                 input_dims,
                 num_actions,
                 fc1_dims=256,
                 fc2_dims=256,
                 name='value',
                 shared=False,
                 dueling_net=False
                 ):
        super().__init__()

        self.input_dims = input_dims
        self.num_actions = num_actions

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(self.input_dims, self.fc1_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc1_dims, self.fc2_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc2_dims, self.num_actions)
            )
        else:
            self.a_head = nn.Sequential(
                nn.Linear(self.input_dims, self.fc1_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc1_dims, self.fc2_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc2_dims, self.num_actions)
            )
            self.v_head = nn.Sequential(
                nn.Linear(self.input_dims, self.fc1_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc1_dims, self.fc2_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc2_dims, 1)
            )

        self.shared = shared
        self.dueling_net = dueling_net

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)

        self.to(self.device)

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(nn.Module):
    def __init__(self,
                 input_dims,
                 num_actions,
                 gpuUse=True,
                 shared=False,
                 dueling_net=False,
                 chkpt_dir='tmp/sac'):
        super().__init__()

        self.gpuUse = gpuUse

        self.Q1 = QNetwork(input_dims,
                           num_actions,
                           fc1_dims=256,
                           fc2_dims=256,
                           name='value',
                           shared=False,
                           dueling_net=False)
        self.Q2 = QNetwork(input_dims,
                           num_actions,
                           fc1_dims=256,
                           fc2_dims=256,
                           name='value',
                           shared=False,
                           dueling_net=False)

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)

        self.Q1.to(self.device)
        self.Q2.to(self.device)

    def forward(self, states):

        states = states.to(self.device)

        q1 = self.Q1(states)
        q2 = self.Q2(states)

        q1 = q1.cpu()
        q2 = q2.cpu()

        return q1, q2

    def save_checkpoint(self):
        torch.save(self.q1.state_dict(), self.checkpoint_file)
        torch.save(self.q2.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))



class ActorNetwork(nn.Module):
    def __init__(self,
                 input_dims,
                 fc1_dims=256,
                 fc2_dims=256,
                 n_actions=2,
                 gpuUse =True,
                 name='actor',
                 chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.gpuUse = gpuUse
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.n_actions)


        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        actionProb = self.fc3(prob)

        return actionProb

    def act(self,state):
        # do greedy action
        actionProbs = self.forward(state)

        return torch.argmax(actionProbs,dim=1,keepdim=True)

    def sample(self, state):
        # do action by probs which proportional to actionProb
        actionProbs = F.softmax(self.forward(state),dim=1)
        actionDist = Categorical(actionProbs)
        actions = actionDist.sample().view(-1,1)

        z = (actionProbs == 0.0).float() * 1e-8

        logActionProbs = torch.log(actionProbs + z)

        return actions, actionProbs , logActionProbs


    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))





































































