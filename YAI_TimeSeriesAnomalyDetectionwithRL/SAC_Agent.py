import os
import torch
import torch.nn.functional as F
import numpy as np
from SAC_Buffer import ReplayBuffer
from SAC_networks import ActorNetwork, TwinnedQNetwork
from torch.optim import Adam
from UTILS import disable_gradients
import datasets.datasetVer1 as theDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from util.usefulFuncs import createDirectory

class Agent():
    def __init__(self,
                 plotSaveDir,
                 input_dims=[8],
                 gpuUse= True,
                 lr=3e-4,
                 gamma=0.99,
                 n_actions=2,
                 max_size=1000000,
                 tau=0.005,
                 targetEntropyRatio=0.98,
                 batch_size=256,
                 reward_scale=2):

        self.plotSaveDir = plotSaveDir
        createDirectory(self.plotSaveDir)
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.gpuUse = gpuUse
        self.scale = reward_scale

        self.targetEntropyRatio = targetEntropyRatio

        self.actor = ActorNetwork(input_dims[0],
                                  n_actions=n_actions,
                                  name='actor'
                                  )

        self.onlineCritic = TwinnedQNetwork(input_dims=input_dims[0],
                                            num_actions=n_actions,
                                            gpuUse=True,
                                            shared=False,
                                            dueling_net=False,
                                            chkpt_dir='tmp/sac'
                                            )

        self.targetCritic = TwinnedQNetwork(input_dims=input_dims[0],
                                            num_actions=n_actions,
                                            gpuUse=True,
                                            shared=False,
                                            dueling_net=False,
                                            chkpt_dir='tmp/sac'
                                            )

        self.targetCritic.load_state_dict(self.onlineCritic.state_dict())

        disable_gradients(self.targetCritic)

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)


        self.policy_optim = Adam(self.actor.parameters(), lr=lr)
        self.q1_optim = Adam(self.onlineCritic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.onlineCritic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.targetEntropy = -np.log(1.0 / self.n_actions)*self.targetEntropyRatio

        # We optimize log(alpha), instead of alpha.
        self.logAlpha = torch.zeros(1, requires_grad=True, device='cpu')
        self.alpha = self.logAlpha.exp()
        self.alpha_optim = Adam([self.logAlpha], lr=lr)

        self.q1LossLst = []
        self.q2LossLst = []
        self.policyLossLst = []
        self.alphaLossLst = []

        self.q1LossLstAvg = []
        self.q2LossLstAVg = []
        self.policyLossLstAvg = []
        self.alphaLossLstAvg = []

        # self.actor.to(self.device)
        # self.onlineCritic.to(self.device)
        # self.targetCritic.to(self.device)
        # self.alpha.to(self.device)

    def explore(self, state):
        # Act with randomness. when training
        with torch.no_grad():
            action, _, _ = self.actor.sample(state)
        return action.item()

    def exploit(self, state):

        # Act without randomness. when validating
        with torch.no_grad():
            action = self.actor.act(state)
        return action.item()

    def remember(self,
                 state,
                 action,
                 reward,
                 new_state,
                 done):

        self.memory.store_transition(state, action, reward, new_state, done)


    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def updateTarget(self):
        self.targetCritic.load_state_dict(self.onlineCritic.state_dict())

    def calcCurrentQ(self,states,actions,rewards,nextStates,dones):

        currQ1, currQ2 = self.onlineCritic(states)

        currQ1 = currQ1.gather(1,actions.long())
        currQ2 = currQ2.gather(1,actions.long())

        return currQ1,currQ2

    def calcTargetQ(self,states,actions,rewards,nextStates,dones):


        with torch.no_grad():
            _,actionProbs,logActionProbs = self.actor.sample(nextStates)
            nextQ1,nextQ2 = self.targetCritic(nextStates)
            # print('Q1 size',nextQ1.size())
            # print('Q1 type',nextQ1.type())
            # print('alpha size,',self.alpha.size())
            # print('alpha type,', self.alpha.type())
            # print('logAction size',logActionProbs.size())
            # print('logAction type ',logActionProbs.type())

            nextQ = (actionProbs *(
                torch.min(nextQ1,nextQ2) - self.alpha * logActionProbs
            )).sum(dim=1,keepdim=True)

        # print(2222222222222222222,nextQ.size())
        # print(dones.size())
        # print(dones)
        # print(3333333333333333333333,rewards.size())
        assert rewards.shape == nextQ.shape
        return rewards + (1.0- dones*1)* self.gamma * nextQ

    def calcCriticLoss(self,batch):

        currQ1,currQ2 = self.calcCurrentQ(*batch)
        targetQ = self.calcTargetQ(*batch)

        errors = torch.abs(currQ1.detach() - targetQ)

        meanQ1= currQ1.detach().mean().item()
        meanQ2 = currQ2.detach().mean().item()

        lossQ1 = torch.mean( (currQ1-targetQ) ).pow(2)
        lossQ2 = torch.mean( (currQ2-targetQ) ).pow(2)

        return lossQ1, lossQ2, errors, meanQ1, meanQ2

    def calcPolicyLoss(self,states,actions,rewards,nextStates,dones):

        _,actionProbs,logActionProbs = self.actor.sample(states)

        with torch.set_grad_enabled(False):
            Q1,Q2 = self.onlineCritic(states)

            Q = torch.min(Q1,Q2)

        entropies = -torch.sum(actionProbs * logActionProbs, dim=1,keepdim=True)

        Q = torch.sum(torch.min(Q1,Q2)*actionProbs,dim=1,keepdim=True)

        policyLoss = ((-Q - self.alpha * entropies)).mean()

        return policyLoss, entropies.detach()

    def calcEntropyLoss(self,entropies):

        assert not entropies.requires_grad

        entropyLoss = -torch.mean(self.logAlpha*(self.targetEntropy-entropies))

        return entropyLoss


    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return

        # self.actor.to(self.device)
        # self.onlineCritic.to(self.device)
        # self.targetCritic.to(self.device)
        # self.alpha.to(self.device)
        #
        # self.onlineCritic.to('cpu')
        # self.actor.to('cpu')
        # self.alpha.to('cpu')
        # # self.targetCritic.to('cpu')
        #
        # print('a is in',self.onlineCritic.device)
        # print('b is in', self.actor.device)
        # print('c is in', self.alpha.device)
        # print('d is in ',self.targetCritic.device)

        BATCH = self.memory.sample_buffer(self.batch_size)

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calcCriticLoss(BATCH)
        policy_loss, entropies = self.calcPolicyLoss(*BATCH)
        entropy_loss = self.calcEntropyLoss(entropies)

        self.updateParams(optim=self.q1_optim,loss=q1_loss)
        self.updateParams(optim=self.q2_optim, loss=q2_loss)
        self.updateParams(optim=self.policy_optim,loss=policy_loss)
        self.updateParams(optim=self.alpha_optim,loss=entropy_loss)

        self.alpha = self.logAlpha.exp()

        # self.onlineCritic.to(self.device)
        # self.actor.to(self.device)
        # self.alpha.to(self.device)
        # self.targetCritic.to(self.device)

        self.q1LossLst.append(q1_loss.detach().item())
        self.q2LossLst.append(q2_loss.detach().item())
        self.policyLossLst.append(policy_loss.detach().item())
        self.alphaLossLst.append(entropy_loss.detach().item())

    def flushLst(self):

        self.q1LossLst.clear()
        self.q2LossLst.clear()
        self.policyLossLst.clear()
        self.alphaLossLst.clear()

        print('flushing lst complete')

    def plotAvgLosses(self):

        self.q1LossLstAvg.append(np.mean(self.q1LossLst))
        self.q2LossLstAVg.append(np.mean(self.q2LossLst))
        self.policyLossLstAvg.append(np.mean(self.policyLossLst))
        self.alphaLossLstAvg.append(np.mean(self.alphaLossLst))

        fig = plt.figure(constrained_layout= True)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(range(len(self.q1LossLstAvg)), self.q1LossLstAvg)
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Q1 Loss')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(range(len(self.q2LossLstAVg)), self.q2LossLstAVg)
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Q2 Loss')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(range(len(self.policyLossLstAvg)), self.policyLossLstAvg)
        ax3.set_xlabel('iteration')
        ax3.set_ylabel('Loss')
        ax3.set_title('Policy Loss')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(range(len(self.alphaLossLstAvg)), self.alphaLossLstAvg)
        ax4.set_xlabel('iteration')
        ax4.set_ylabel('Loss')
        ax4.set_title('Alpha Loss')

        plt.savefig(self.plotSaveDir+'trainingLossResult.png', dpi=200)
        print('saving plot complete!')
        plt.close()
        plt.cla()
        plt.clf()

        self.flushLst()

    def updateParams(self,optim,loss):

        optim.zero_grad()
        loss.backward()
        optim.step()












