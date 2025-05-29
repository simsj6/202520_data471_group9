# Predicting the beer style from the specific ingredients (e.g., hops, yeasts, or fermentables) used to brew the beer. This is also a classification task.

import argparse
import numpy as np
import sklearn
from scipy.sparse import coo_array

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-config', type=str, required=True,
                        help='Task config file')
    parser.add_argument('-train_feat', type=str, required=True,
                        help='Training set feature file')
    parser.add_argument('-train_target', type=str, required=True,
                        help='Training set target file')
    parser.add_argument('-dev_feat', type=str, required=True,
                        help='Development set feature file')
    parser.add_argument('-dev_target', type=str, required=True,
                        help='Development set target file')
    # parser.add_argument('-nunits', type=int, required=True,
    #                     help='Number of hidden units per layer')
    # parser.add_argument('-nlayers', type=int, required=True,
    #                     help='Number of hidden layers')
    # parser.add_argument('-hidden_act', type=str, required=True, choices=['sig', 'tanh', 'relu'],
    #                     help='Hidden unit activation function')
    # parser.add_argument('-type', type=str, required=True, choices=['C', 'R'],
    #                     help='Problem mode: C for classification, R for regression')
    # parser.add_argument('-output_dim', type=int, required=True,
    #                     help='Number of classes or output dimension')
    # parser.add_argument('-total_updates', type=int, required=True,
    #                     help='Total number of updates (gradient steps)')
    # parser.add_argument('-learnrate', type=float, required=True,
    #                     help='Learning rate')
    # parser.add_argument('-init_range', type=float, required=True,
    #                     help='Range for uniform random initialization')
    # parser.add_argument('-mb', type=int, required=True,
    #                     help='Minibatch size (0 for full batch)')
    # parser.add_argument('-report_freq', type=int, required=True,
    #                     help='Frequency of reporting performance')
    # parser.add_argument('-v', action='store_true',
    #                     help='Verbose mode')
    
    return parser.parse_args()

def format_data(path, shape):
    rows = []
    cols = []
    data = []

    with open(path, 'r') as file:
        for line in file:
            row, col, value = map(float, line.strip().split())
            rows.append(row)
            cols.append(col)
            data.append(value)
    
    array = coo_array((data, (rows, cols)), shape=shape).toarray()
    return array

def main():
    # parse arguments
    args = parse_arguments()
    config = args.config
    trainfeat_fn = args.train_feat
    traintarget_fn = args.train_target
    devfeat_fn = args.dev_feat
    devtarget_fn = args.dev_target
    # numupdates = args.total_updates
    # mode = args.type
    # numlayers = args.nlayers
    # numunits = args.nunits
    # actfunc = args.hidden_act
    # initrange = args.init_range
    # outputdim = args.output_dim
    # learnrate = args.learnrate
    # reportfreq = args.report_freq
    # isverbose = args.v

    # get details about data from config file
    with open(config) as file:
        token = file.read().split()
    
    n_train = int(token[1])
    n_dev = int(token[3])
    d = int(token[5])
    c = int(token[7])

    # create arrays from data
    # training features
    trainfeat_shape = (n_train, d)
    trainfeat_array = format_data(trainfeat_fn, trainfeat_shape)

    # training targets
    traintarget_array = np.loadtxt(traintarget_fn)

    # dev features
    devfeat_shape = (n_dev, d)
    devfeat_array = format_data(devfeat_fn, devfeat_shape)

    # dev targets
    devtarget_array = np.loadtxt(devtarget_fn)

if __name__ == "__main__":
    main()
