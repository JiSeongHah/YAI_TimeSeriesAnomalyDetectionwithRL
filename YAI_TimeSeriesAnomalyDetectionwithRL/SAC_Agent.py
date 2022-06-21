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


class Agent():
    def __init__(self,
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

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.gpuUse = gpuUse
        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.targetEntropyRatio = targetEntropyRatio


        self.actor = ActorNetwork(input_dims,
                                  n_actions=n_actions,
                                  name='actor'
                                  )

        self.onlineCritic = TwinnedQNetwork(input_dims=input_dims,
                                            num_actions=n_actions,
                                            gpuUse=True,
                                            shared=False,
                                            dueling_net=False,
                                            chkpt_dir='tmp/sac'
                                            )

        self.targetCritic = TwinnedQNetwork(input_dims=input_dims,
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
        self.logAlpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.logAlpha.exp()
        self.alpha_optim = Adam([self.logAlpha], lr=lr)

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

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

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

    def calcCurrentQ(self,states,actions,rewards,nextStates,donees):

        currQ1, currQ2 = self.onlineCritic(states)

        currQ1 = currQ1.gather(1,actions.long())
        currQ2 = currQ2.gather(1,actions.long())

        return currQ1,currQ2

    def calcTargetQ(self,states,actions,rewards,nextStates,dones):

        with torch.set_grad_enabled(False):
            _,actionProbs,logActionProbs = self.actor.sample(nextStates)

            nextQ1,nextQ2 = self.targetCritic(nextStates)

            nextQ = (actionProbs *(
                torch.min(nextQ1,nextQ2) - self.alpha * logActionProbs
            )).sum(dim=1,keepDim=True)

        assert rewards.shape == nextQ.shape
        return rewards + (1.0- dones)* self.gamma * nextQ

    def calcCriticLoss(self,batch,weights):

        currQ1,currQ2 = self.calcCurrentQ(*batch)
        targetQ = self.calcTargetQ(*batch)

        errors = torch.abs(currQ1.detach() - targetQ)

        meanQ1= currQ1.detach().mean().item()
        meanQ2 = currQ2.detach().mean().item()

        lossQ1 = torch.mean( (currQ1-targetQ) ).pow(2) * weights
        lossQ2 = torch.mean( (currQ2-targetQ) ).pow(2) * weights

        return lossQ1, lossQ2, errors, meanQ1, meanQ2

    def calcPolicyLoss(self,states,actions,rewards,nextStates,dones,weights):

        _,actionProbs,logActionProbs = self.actor.sample(states)

        with torch.set_grad_enabled(False):
            Q1,Q2 = self.onlineCritic(states)

            Q = torch.min(Q1,Q2)

        entropies = -torch.sum(actionProbs * logActionProbs, dim=1,keepdim=True)

        Q = torch.sum(torch.min(Q1,Q2)*actionProbs,dim=1,keepdim=True)

        policyLoss = (weights*(-Q - self.alpha * entropies)).mean()

        return policyLoss, entropies.detach()

    def calcEntropyLoss(self,entropies,weights):

        assert not entropies.requires_grad

        entropyLoss = -torch.mean(self.logAlpha*(self.targetEntropy-entropies)*weights)

        return entropyLoss


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        BATCH = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()












