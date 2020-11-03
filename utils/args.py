import argparse

DATASETS = ['cifar-100-python']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='directory of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='cifar-100-python')
    parser.add_argument('-bin_model',
                    help='directory of binary model;',
                    type=str,
                    default='bin-logs')
    
    parser.add_argument('-non_bin_model',
                    help='derectory of full accuracy model;',
                    type=str,
                    default='bin-weights')

    parser.add_argument('--eval_every_n',
                    help='validate model every _____ steps',
                    type=int,
                    default=100)
    
    parser.add_argument('--max_steps',
                help='max steps',
                type=int,
                default=5000)
    
    parser.add_argument('-resize_image',
                dest = "resize",
                help='resize image',
                type=bool,
                default=False)
                
    return parser.parse_args()
                                    