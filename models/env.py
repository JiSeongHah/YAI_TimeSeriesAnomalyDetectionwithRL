import torch



class Yahoo_ENV:
    def __init__(self,state_set,label_set,args):
        self.state = state_set
        self.label = label_set
        self.len = len(self.label)

        self.tp = args.TP
        self.fp = args.FP
        self.fn =args.FN
        self.tn = args.TN
        
    def reset(self): #initial state
        return self.state[0]
    
    def step(self,action,idx): 
        a_pred = action
        a_real = self.label[idx]
        reward = self.get_reward(a_pred,a_real)

        if self.len == idx:
            done = True
        else:
            done = False
            next_state = self.state[idx+1]        
        return next_state,reward, done

    def get_reward(self,action_pred,action_real):
        if action_real == 1: #anomaly
            if action_pred == 1: # TP
                return self.tp
            elif action_pred == 0: #FP
                return self.fn
        elif action_real == 0: #not anomaly
            if action_pred == 1: #FN
                return self.FN
            if action_pred == 0: #TN
                return self.TN
    
    def render(self,reward):
        self.reward.append(reward)
    
