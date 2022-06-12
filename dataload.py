from config import get_parse
from datasets.Yahoo import build_yahoo

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
    
    # print("if A1 DATASET,")
    # for i, real_i in enumerate(train):
    #     ts, val, label = real_i
    #     print(f"real_{str(i)}timestamp:{ts},{ts.shape}")
    #     print(f"value:{val},{val.shape}")
    #     print(f"label:{label}.{label.shape}")
    return train, test


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
            