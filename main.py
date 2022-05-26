from config import get_parse
from datasets.Yahoo import build_yahoo

def main(args):
    
    if args.datasets == 'Yahoo':
        train_loader ,test_loader = build_yahoo(args)
        
    elif args.dataset == 'SWaT':
        pass 

    elif args.dataset == 'KPI':
        pass

    elif args.dataset == 'Numenta':
        pass
    
    for time_stamp, value, label in (train_loader):
        print(f'time_stamp : {time_stamp}')
        print(f'value : {value},{value.size}')
        print(f'label : {label},{label.size()}')
        print('-'*50)



if __name__ == '__main__': 
    args = get_parse()
    main(args)





