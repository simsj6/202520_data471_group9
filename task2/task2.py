# Name: Jenny Sims
# Date: 06/05/25
# Description: Predicting the number of strikeouts a baseball player will have in the next 10 games given the number of walks, singles, doubles, triples and homeruns in each of his last 20 games. This is a regression task.

# example call:
# python task2.py -train_feat train.X -train_target train.RT -dev_feat dev.X -dev_target dev.RT

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-train_feat', type=str, required=True,
                        help='Training set feature file')
    parser.add_argument('-train_target', type=str, required=True,
                        help='Training set target file')
    parser.add_argument('-dev_feat', type=str, required=True,
                        help='Development set feature file')
    parser.add_argument('-dev_target', type=str, required=True,
                        help='Development set target file')
    
    return parser.parse_args()

def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        hidden_layer = 8
        self.hidden_linear1 = torch.nn.Linear(n, hidden_layer)
        self.hidden_linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        self.hidden_linear3 = torch.nn.Linear(hidden_layer, hidden_layer)
        self.output_linear = torch.nn.Linear(hidden_layer, 1)

    def forward(self, x):
        a1 = torch.nn.functional.relu(self.hidden_linear1(x))
        a2 = torch.nn.functional.relu(self.hidden_linear2(a1))
        a3 = torch.nn.functional.relu(self.hidden_linear2(a2))
        y_pred = self.output_linear(a3)
        return y_pred
    
def initial_model(X_train, y_train, X_dev, y_dev):
    X = torch.Tensor(standardize(X_train))
    y = torch.Tensor(standardize(y_train))

    dev_X = torch.Tensor(standardize(X_dev))
    dev_y = torch.Tensor(standardize(y_dev))

    # train_model = torch.nn.Linear(train_components, 1)
    model = Model(X_train.shape[1])
    # model = torch.nn.Linear(X_train.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Instantiate optimizer object
    
    for i in range(1000):
        y_pred = model(X).squeeze()

        # Using torch's MSE
        error = torch.nn.functional.mse_loss(y_pred, y)

        # Calculate gradients
        error.backward()
        optimizer.step() # Update parameters
        optimizer.zero_grad() # Clear stored gradients
        print(f"\rUpdate: {i + 1}\tError: {error}", end="", flush=True)
    print()

    # having low training error gives high dev error :) i love that :)
    y_pred_dev = model(dev_X).squeeze()
    dev_error = torch.nn.functional.mse_loss(y_pred_dev, dev_y)
    print(f"Dev Error: {dev_error}")

def main():
    # parse arguments
    args = parse_arguments()
    trainfeat_fn = args.train_feat
    traintarget_fn = args.train_target
    devfeat_fn = args.dev_feat
    devtarget_fn = args.dev_target

    # create arrays from data
    # training features
    X_train = np.loadtxt(trainfeat_fn)
    y_train = np.loadtxt(traintarget_fn)

    # dev features
    X_dev = np.loadtxt(devfeat_fn)
    y_dev = np.loadtxt(devtarget_fn)

    # print("arrays made", flush=True)

    # model time 🥴
    print("it's modelin' time") # and model all over the place
    initial_model(X_train, y_train, X_dev, y_dev)

if __name__ == "__main__":
    main()