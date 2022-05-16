import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self,args):
        super(Model).__init__()

    def forward(self):
        return 
        
class ICMModel(nn.Module):
    def __init__(self, args):
        super(ICMModel, self).__init__()

        self.input_size = args.input_size
        self.output_size = 2
        self.device = torch.device(args.device)

        self.embeddings = nn.Embedding(2, emddim)
        self.lstm = nn.LSTM(EMBDDIM, HIDDENDIM)
        
        feature_output = 

        self.encoder = #TODO
        

        self.inverse_net = #TODO

        self.residual = #TODO

        self.forward_net_1 = #TODO

        self.forward_net_2 = #TODO, prediction of next state


    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state = self.encoder(state)
        encode_next_state = self.encoder(next_state)
        
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state = encode_next_state
        return real_next_state, pred_next_state, pred_action