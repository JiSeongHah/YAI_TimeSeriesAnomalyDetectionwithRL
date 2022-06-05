import torch
import torch.nn as nn

import numpy as np


import torch
import torch.nn.functional as F

def total_loss(args,agent,loss_fns,replay):
    
    q_loss_func,f_loss_func,i_loss_func= loss_fns # loss functions unpack

    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
    action_batch = action_batch.view(action_batch.shape[0],1)
    reward_batch = reward_batch.view(reward_batch.shape[0],1)

    #loss

    
    #reward
    i_reward = agent.get_intrinsic_reward(state1_batch, action_batch, state2_batch)
    reward = i_reward.detach()
    reward += reward_batch 
    qvals = agent.get_action(state2_batch,get_qval = True)
    reward += args.gamma * torch.max(qvals)

    reward_pred = agent.get_action(state1_batch,get_qval = True)
    reward_target = reward_pred.clone()
    
    indices = torch.stack( (torch.arange(action_batch.shape[0]),action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), F.normalize(reward_target.detach()))

    loss = loss_sum(q_loss, forward_pred_err, inverse_pred_err)

    return loss

def loss_sum(args, q_loss, forward_loss, inverse_loss):
    loss = args.lambda_ * q_loss + (1 - args.beta_) * inverse_loss + args.beta_ * forward_loss 
    return loss

