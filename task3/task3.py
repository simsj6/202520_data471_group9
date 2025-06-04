# Predicting the rating a user gives a product based on the term-frequency vector of their review of the product. This is also a regression task.
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings
import torch

# python task3.py -config "./task3_ratings/task3.config" -train_feat "./task3_ratings/train.sparseX" -train_target "./task3_ratings/train.RT" -dev_feat "./task3_ratings/dev.sparseX" -dev_target "./task3_ratings/dev.RT"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, 
                        help='Task config file')
    parser.add_argument('-train_feat', type=str, required=True, 
                        help='Training feature file')
    parser.add_argument('-train_target', type=str, required=True, 
                        help='Training target file')
    parser.add_argument('-dev_feat', type=str, required=True, 
                        help='Development feature file')
    parser.add_argument('-dev_target', type=str, required=True, 
                        help='Development target file')
    return parser.parse_args()

# Load config values
def load_config(path):
    with open(path, 'r') as f:
        token = f.read().split()
    n_train = int(token[1])
    n_dev = int(token[3])
    d = int(token[5])
    c = int(token[7])
    return n_train, n_dev, d, c

# Load sparse data as sparse matrix
def load_sparse(path, shape):
    rows, cols, vals = [], [], []
    with open(path, 'r') as file:
        for line in file:
            lineList = list(map(float, line.strip().split()))
            # row, col, value = map(float, line.strip().split())
            row = lineList[0]
            col = lineList[1]
            value = lineList[2]
            rows.append(int(row))
            cols.append(int(col))
            vals.append(value)
    return coo_matrix((vals, (rows, cols)), shape=shape)

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

    y_pred_dev = model(dev_X).squeeze()
    dev_error = torch.nn.functional.mse_loss(y_pred_dev, dev_y)
    print(f"Dev Error: {dev_error}")

def main():
    # parse arguments
    args = parse_arguments()
    config_fn = args.config
    trainfeat_fn = args.train_feat
    traintarget_fn = args.train_target
    devfeat_fn = args.dev_feat
    devtarget_fn = args.dev_target

    # Load data
    n_train, n_dev, d, c = load_config(args.config)
    
    X_train = load_sparse(args.train_feat, (n_train, d)).tocsr()
    y_train = np.loadtxt(args.train_target, dtype=int)

    X_dev = load_sparse(args.dev_feat, (n_dev, d)).tocsr()
    y_dev = np.loadtxt(args.dev_target, dtype=int)

    print("Data loaded.")

    # Suppress convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("arrays made", flush=True)

    print("Running model")
    initial_model(X_train, y_train, X_dev, y_dev)

if __name__ == "__main__":
    main()