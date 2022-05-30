import numpy as np


def sliding_window(dataset : list,args):
        
        ws = args.window_size

        if args.datasets == 'Yahoo':
            
            state = {} #OUTPUT; consist of timestamp, value, label
            time_state =[]  # sliding_window
            value_state = []
            label_state = []

            for data_i in dataset: # real_i or synthetic_i : set
                ts = np.array(data_i['timestamp']) #list
                val = np.array(data_i['value']) 
                label = np.array(data_i['label'])
                
                num_samples = len(ts) - ws + 1
                for j in range(num_samples):
                    ts_ = ts[j:j+ws]   #list
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

        elif args.datasets == 'Numenta':
            pass
        
        elif args.datasets == 'KPI':
            pass