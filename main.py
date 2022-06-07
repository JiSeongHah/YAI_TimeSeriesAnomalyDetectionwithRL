from config import get_parse
from datasets.Yahoo import build_yahoo

def main(args):
    
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
    for i, real_i in enumerate(train):
        ts, val, label = real_i
        print(f"real_{str(i)}timestamp:{ts},{ts.shape}")
        print(f"value:{val},{val.shape}")
        print(f"label:{label}.{label.shape}")



if __name__ == '__main__': 
    args = get_parse()
    main(args)





