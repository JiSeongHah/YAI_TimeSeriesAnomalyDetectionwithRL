#PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np
from config import get_parse
from dataload import get_nth_data, main_data
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

#Hyperparameters
learning_rate = 0.001
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
beta_val = 1
cnt_TP = 0
cnt_TN = 0
cnt_FP = 0
cnt_FN = 0

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        #self.fc1   = nn.Linear(4,64)
        #self.lstm  = nn.LSTM(64,32)
        #self.fc_pi = nn.Linear(32,2)
        #self.fc_v  = nn.Linear(32,1)
        self.fc1   = nn.Linear(50,64)
        self.lstm  = nn.LSTM(64,32)
        self.fc_pi = nn.Linear(32,2)
        self.fc_v  = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

def reward(true, pred, cnt_TP, cnt_TN, cnt_FP, cnt_FN):
    TP = 2.001 #-0.05 #0.156
    TN = -1.003 #0.002
    FP = -1.2 #-0.0001
    FN = 1.1 #-1.05
    # 10, 1, -10, -3 & beta = 10e-1-> 나쁘지 않음
    # 20, -10, -15, 1 & beta = 1-> 2개만 나옴
    # 17, -10, -15, 2 & beta = 1-> 값 작게 나옴
    # 15, -8, -10, 2 & beta = 1-> 값 작게 나옴
    # 0.9, 0.01, -0.001, -0.3 & beta = 1 -> 값 작게 나옴
    if(true == 1 and pred == 1):
        cnt_TP = cnt_TP + 1
        return TP, cnt_TP, cnt_TN, cnt_FP, cnt_FN
    elif (true == 1 and pred == 0):
        cnt_FN = cnt_FN + 1
        return TN, cnt_TP, cnt_TN, cnt_FP, cnt_FN
    elif (true == 0 and pred == 1):
        cnt_FP = cnt_FP + 1
        return FP, cnt_TP, cnt_TN, cnt_FP, cnt_FN
    elif (true == 0 and pred == 0):
        cnt_TN = cnt_TN + 1
        return FN, cnt_TP, cnt_TN, cnt_FP, cnt_FN

#def rewards(true, pred):
def cal_fbetascore(y_true, y_pred, beta):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if((beta**2 * precision + recall) == 0):
        fsc = 0
    else:
        fsc = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    return precision, recall, fsc

def main_ppo_lstm():
    ######################## make environment ########################
    #env = gym.make('CartPole-v1')
    #model = PPO()
    #score = 0.0
    #print_interval = 20
    args = get_parse()
    train, test = main_data(args)
    model = PPO()
    score = 0.0
    num_epi = len(train.timestamp)
    #model.train_net()
    ##################################################################
    print('-'*80)
    
    print(num_epi)
    for n_epi in range(num_epi):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        time_arr = train.timestamp[n_epi]
        val_arr = train.value[n_epi]
        label_arr = train.label[n_epi]
        s = time_arr[0]
        done = False
        pred_lst = []
        true_lst = []
        cnt_TP = 0
        cnt_TN = 0
        cnt_FP = 0
        cnt_FN = 0

        while not done:
            for t in range(len(time_arr)):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob) # 맞나????????
                a = m.sample().item()
                #s_prime, r, done, info = env.step(a)
                s_prime = time_arr[t+1]
                r, cnt_TP, cnt_TN, cnt_FP, cnt_FN = reward(label_arr[t], a, cnt_TP, cnt_TN, cnt_FP, cnt_FN)
                pred_lst.append(a)
                true_lst.append(label_arr[t])
                if t == len(time_arr)-2:
                    done = True
                else:
                    done = False
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime

                score += r
                if done:
                    break
                    
            model.train_net()

        #confusion matrix : TN, FP \ FN, TP
        #print(confusion_matrix(true_lst, pred_lst)) # hyperparameter tuning -> 유전알고리즘 이용해서 자동화도 됨, 오래걸리긴 함 / 중요도 낮은 hyperparameter는 크게 안바꿔도
        #print('cnt_TN, cnt_FP, cnt_FN, cnt_TP : ', cnt_TN, cnt_FP, cnt_FN, cnt_TP)
        precision, recall, f_beta_score = cal_fbetascore(true_lst, pred_lst, beta_val) # 분모에 작은 값 넣기 / if문써서 분모 0이면 다른 값 출력하도록 (분모 0인경우에는 !score 0!이거나 계산안하도록) // 논문 beta // reward, hyperparameter 튜닝

        print("precision :{:.4f}, recall :{:.4f}, f_score :{:.4f}".format(precision, recall, f_beta_score))
        #print("# of data :{}, length :{}, avg score : {:.1f}, f_score :{:.3f}".format(n_epi, len(time_arr), score, f_beta_score))
        score = 0.0

    print('-'*80)
    #env.close()

if __name__ == '__main__':
    main_ppo_lstm()