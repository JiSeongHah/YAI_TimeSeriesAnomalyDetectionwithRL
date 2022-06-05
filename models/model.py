import torch
import torch.nn as nn
import torch.nn.functional as F


'''
dimension 맞ㄹ추기
'''

class ICMagent:

    def __init__(self,args):

        self.device = args.device

        self.icm = ICMModel(args)
        self.qval = Qnetwork(args)

        self.eps = args.eps

    
    def get_action(self,state,get_qval = False):

        state = torch.Tensor(state).to(self.device)
        state = state.float()
        qvals = self.qval(state)    
        action = epsilon_greedy_policy(qvals,eps = self.eps)

        if get_qval:
            return qvals    
        else:
            return action # 0 or 1
    
    
    def get_intrinsic_reward(self,state,action,next_state):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(
            len(action), self.output_size).to(
            self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action_onehot])
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
        return intrinsic_reward.data.cpu().numpy()
        
class ICMModel(nn.Module):
    def __init__(self, args):
        super(ICMModel, self).__init__()

        self.device = torch.device(args.device)

        self.encoder = nn.LSTM(1,64,3,batch_first = True) 

        self.inverse_net = nn.Sequential(
            nn.Linear(64* 2,512),
            nn.ReLU(),
            nn.Linear(512,2)
        )

        self.forward_net = nn.Sequential(
            nn.Linear(2 + 512 ,512),
            nn.LeakyReLU()
        )

    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state, _ = self.encoder(state)

        encode_next_state, _ = self.encoder(next_state)
        
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        
        # get pred next state
        pred_next_state = torch.cat((encode_state, action), 1)
        pred_next_state = self.forward_net(pred_next_state)

        real_next_state = encode_next_state
        return real_next_state, pred_next_state, pred_action

class Qnetwork(nn.Module):
    def __init__(self,args):
        super(Qnetwork, self).__init__()
        self.lstm = nn.LSTM(1,2,3,batch_first= True)

    def forward(self,x):
        outputs, _ = self.lstm(x)
        action = outputs[:,-1]
        return action #batch,2


def epsilon_greedy_policy(qvals,eps):
    if torch.rand(1) < eps:
        return torch.randint(low=0,high=1,size=(1,)) # 0 : normal, 1 : anomaly
    else:
        return torch.argmax(qvals)