import numpy as np


def sliding_window(dataset : list,args):
        
        ws = args.window_size

        if args.datasets == 'Yahoo':
            state = {}

            time_state =[]
            value_state = []
            label_state = []

            for i, data_i in enumerate(dataset): # data_i : set
            
                ts = np.array(data_i['timestamp']) #list
                val = np.array(data_i['value']) 
                label = np.array(data_i['label'])

                num_samples = len(ts) - ws + 1

                for j in range(num_samples):

                    ts_ = ts[j:j+ws]    
                    val_ = val[j:j+ws]
                    lab_ = label[j:j+ws]

                    time_state.append(ts_)
                    value_state.append(val_)
                    label_state.append(lab_)
            state['timestamp'] = time_state
            state['value'] = value_state
            state['label'] = label_state      

            return state

        elif args.datasets == 'SWaT':
            pass