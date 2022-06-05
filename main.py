import torch
import torch.nn as nn
import torch.optim


from config import get_parse
from datasets.Yahoo import build_yahoo
from models.env import Yahoo_ENV
from models.loss import total_loss
from models.model import ICMagent
from util.ExperienceReplay import ExperienceReplay

def main(args):
    
    if args.datasets == 'Yahoo':
        train,test = build_yahoo(args)
        
    elif args.dataset == 'SWaT':
        pass 

    elif args.dataset == 'KPI':
        pass

    elif args.dataset == 'Numenta':
        pass
    
    #init
    agent = ICMagent(args)
    replay = ExperienceReplay(args)
    
    #loss and optim
    f_loss_func = nn.MSELoss()
    i_loss_func = nn.CrossEntropyLoss()
    q_loss_func = nn.MSELoss()
    loss_fns = (q_loss_func,f_loss_func,i_loss_func)
    optim = torch.optim.Adam()

    num_episodes = len(train) #133
    for e in range(num_episodes):
        _, state_set, label_set = train[e]
        env = Yahoo_ENV(state_set,label_set,args)
        
        state= env.reset() #get initial state
        for i in range(args.epochs):
            optim.zero_grad()
            
            action = agent.get_action(state)
            next_state, reward, done = env.step(action,i)
            if done:
                break
            replay.add_memory(state,action,reward,next_state)
            print('no error')
            loss = total_loss(args,agent,loss_fns,replay)
            loss.backward()
            optim.step()

            state = next_state  
    # metric
    # for i in range(5000):
    #     env = Yahoo_ENV
    #     state1 = env.reset()
    #     action - agent.get_action(state1)
    #     state2, reward, done, info = env.step(action)
    #     state2 = prepare_multi_state(state1,state2)
    #     state1=state2
    #     env.render()
    
    # env.plot()


            
if __name__ == '__main__': 
    args = get_parse()
    main(args)