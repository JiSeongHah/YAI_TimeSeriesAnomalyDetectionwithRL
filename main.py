from config import get_parse
from train import Train

def main(args):
    
    if args.datasets == 'Yahoo':
        train_loader ,test_loader = build_yahoo(args)

    elif args.dataset == 'SWaT':
        pass 

    elif args.dataset == 'KPI':
        pass

    elif args.dataset == 'Numenta':
        pass
    
    

if __name__ == '__main__':
    args = get_parse()
    main(args)





