from config import get_parse
from datasets.Yahoo import build_yahoo
from imblearn.over_sampling import SMOTE
import numpy as np

def main_data(args):
    
    if args.datasets == 'Yahoo':
        train ,test = build_yahoo(args)
        
    elif args.dataset == 'SWaT':
        pass 

    elif args.dataset == 'KPI':
        pass

    elif args.dataset == 'Numenta':
        pass
    # if A1
    print("if A1 DATASET,")
    #print(train.timestamp)
    #print('(prev) len_ts, len_val :', len(train.timestamp), len(train.value))
    #print('len_timestamp : ', len(train.timestamp[48]))
    #print('len_value : ', len(train.value[48]))
    for i, real_i in enumerate(train):
        ts, val, label = real_i
        #if (i==48):
            #print(ts)
            # print(type(ts)) # numpy array
            # print(type(ts[0])) # numpy array
            # print(ts.shape) # (1412, 50)
            # print(val.shape) # (1412, 50)
            # print(label.shape) # (1412,)
        #print('len_ts, len_val, len_label : ', len(ts), len(val), len(label))
        #print('shape_Val : ', val.shape)
        #print('shape_Label : ', label.shape)
        #print(list(label).count(1))
        if(list(label).count(1)>6):
            smote = SMOTE(k_neighbors=6) #(sampling_strategy='auto', k_neighbors=k, random_state=seed)
            val, label = smote.fit_sample(val, label) 
            
            train.timestamp[i] = timestamp_sw(ts, len(val))
            train.value[i] = val
            train.label[i] = label
            #if(i==48):
                #print('len_ts, len_val, len_label : ', len(ts), len(val), len(label))
                #print(train.timestamp[i])
        elif(1<list(label).count(1)<=6):
            smote = SMOTE(k_neighbors=list(label).count(1)-1) #(sampling_strategy='auto', k_neighbors=k, random_state=seed)
            val, label = smote.fit_sample(val, label) 
            
            train.timestamp[i] = timestamp_sw(ts, len(val))
            train.value[i] = val
            train.label[i] = label
    #print('len_timestamp : ', len(train.timestamp[48]))
    #print('len_value : ', len(train.value[48]))
    #print('(next) len_ts, len_val :', len(train.timestamp), len(train.value))
    
    # print("if A1 DATASET,")
    # for i, real_i in enumerate(train):
    #     ts, val, label = real_i
    #     print(f"real_{str(i)}timestamp:{ts},{ts.shape}")
    #     print(f"value:{val},{val.shape}")
    #     print(f"label:{label}.{label.shape}")
    return train, test

def timestamp_sw(ts, length_val):
    #times = np.arange(1, length_val+49)
    ts_array = np.full((length_val, 50), 0)
    for i in range(0, length_val):
        ts_array[i,:] = np.arange(i+1, i+51)
    return ts_array

def get_nth_data(index):
    args = get_parse()
    train, test = main_data(args)
    for i, real_i in enumerate(train):
        #print(i)
        if (i==index):
            ts, val, label = real_i
            # print(f"real_{str(i)}timestamp:{ts},{ts.shape}")
            # print(f"value:{val},{val.shape}")
            # print(f"label:{label}.{label.shape}")
            # print("-"*80)
            time_arr = ts
            val_arr = val
            label_arr = label
    return time_arr, val_arr, label_arr

if __name__ == '__main__': 
    args = get_parse()
    train, test = main_data(args)
    #ts, val, label = train
    print(len(train.timestamp[0]))
    print(len(train.value[0]))
    print(len(train.label[0]))
    print(len(train.timestamp))
    print(len(train.value))
    print(len(train.label))
    #'''
    #for i, real_i in enumerate(train):
        #if (i==1):
            #ts, val, label = real_i
            # print(f"real_{str(i)}timestamp:{ts},{ts.shape}")
            # print(f"value:{val},{val.shape}")
            # print(f"label:{label}.{label.shape}")
            # print("-"*80)
            