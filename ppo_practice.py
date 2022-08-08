#from ossaudiodev import SNDCTL_MIDI_PRETIME
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from config import get_parse
from dataload import get_nth_data, main_data
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20  ## 몇 time step 동안 data 모을지
beta          = 10**(-5)
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        # self.fc1   = nn.Linear(4,256)
        # self.fc_pi = nn.Linear(256,2)
        # self.fc_v  = nn.Linear(256,1)
        self.fc1   = nn.Linear(50, 256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0): ## s_t에 대해 inference -> dim = 0 // [s_1, ..., s_t]에 대해 -> dim = 1
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch): ## GAE (generalized advantage estimation) 이용
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            ## cliped loss
            pi = self.pi(s, softmax_dim=1)
            a = a.type(torch.int64)
            pi_a = pi.gather(1,a) ## error 해결 <-gather(): Expected dtype int64 for index
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def reward(pred, true):
    TP = 3
    TN = 1
    FP = 2
    FN = -1
    if(pred == 1 and true == 1):
        return TP
    elif (pred == 1 and true == 0):
        return TN
    elif (pred == 0 and true == 1):
        return FP
    elif (pred == 0 and true == 0):
        return FN

def cal_fbetascore(y_pred, y_true, beta):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fsc = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    return fsc

def main_ppo(): # 빠른 버전
    ######################## make environment ########################
    args = get_parse()
    train, test = main_data(args)
    model = PPO()
    score = 0.0
    num_epi = len(train.timestamp)
    ##################################################################
    print('-'*80)
    for n_epi in range(num_epi): #10000
        time_arr = train.timestamp[n_epi]
        val_arr = train.value[n_epi]
        label_arr = train.label[n_epi]
        s = time_arr[0]
        done = False
        #pred_arr = np.array([])
        #true_arr = np.array([])
        pred_lst = []
        true_lst = []
        while not done:
            for t in range(len(time_arr)): #T_horizon ## T_horizon step 만큼 data 모으고 학습진행 -> 똑똑해진 policy가 다음 step 만큼 진행 
                #print(len(time_arr))
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                #s_prime, r, done, info = env.step(a)
                s_prime = time_arr[t+1]
                r = reward(label_arr[t], a)
                #pred_arr = np.append(pred_arr, a)
                #true_arr = np.append(true_arr, label_arr[t])
                pred_lst.append(a)
                true_lst.append(label_arr[t])
                #print(r)
                if(r==None):
                    print(label_arr[t])
                    print(a)
                    print(reward(label_arr[t], a))
                    print(r)
                if t == len(time_arr)-2:
                    done = True
                else:
                    done = False

                model.put_data((s, a, r, s_prime, prob[a].item(), done)) ## prob[a].item() : action에 대한 확률값 -> ratio 계산할 때 사용
                s = s_prime
                score += r
                if done:
                    break

            model.train_net()

        # if n_epi%print_interval==0 and n_epi!=0:
        #     print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        #     score = 0.0
        f_beta_score = cal_fbetascore(pred_lst, true_lst, beta)
        print("# of episode :{}, length :{}, avg score : {:.1f}, f_score :{:.3f}".format(n_epi, len(time_arr), score, f_beta_score))
        score = 0.0
        #print(fbeta_score(pred_lst, true_lst, average = 'macro', beta = 0.5))
        #print(fbeta_score(pred_lst, true_lst, average = 'micro', beta = 1))
        #print(fbeta_score(pred_lst, true_lst, average = 'weighted', beta = 1))
        #print(fbeta_score(pred_lst, true_lst, average = None, beta = 1))

    print('-'*80)

######## 속도 느림 ########
def main_ppo_1(): 
    ######################## make environment ########################
    model = PPO()
    score = 0.0
    print('-'*80)
    ##################################################################
    for n_epi in range(10): #10000
        time_arr, val_arr, label_arr = get_nth_data(n_epi)
        #s = env.reset() 
        s = time_arr[0]
        done = False
        while not done:
            for t in range(len(time_arr)): #T_horizon ## T_horizon step 만큼 data 모으고 학습진행 -> 똑똑해진 policy가 다음 step 만큼 진행 
                #print(len(time_arr))
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                #s_prime, r, done, info = env.step(a)
                s_prime = time_arr[t+1]
                r = reward(label_arr[t], a)
                #print(r)
                if(r==None):
                    print(label_arr[t])
                    print(a)
                    print(reward(label_arr[t], a))
                    print(r)
                if t == len(time_arr)-2:
                    done = True
                else:
                    done = False
                


                model.put_data((s, a, r, s_prime, prob[a].item(), done)) ## prob[a].item() : action에 대한 확률값 -> ratio 계산할 때 사용
                s = s_prime
                score += r
                if done:
                    break

            model.train_net()

        # if n_epi%print_interval==0 and n_epi!=0:
        #     print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        #     score = 0.0
        print("# of episode :{}, length :{}, avg score : {:.1f}".format(n_epi, len(time_arr), score))
        score = 0.0
    print('-'*80)

####### pangyou 버전 #######
def main_orig(): 
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main_ppo()
    #main_ppo_1()
    #main_orig()