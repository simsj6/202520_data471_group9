# Name: Jenny Sims
# Date: 06/05/25
# Description: Predicting the beer style from the specific ingredients (e.g., hops, yeasts, or fermentables) used to brew the beer. This is a classification task.

# example call:
# python task5.py -config task5.config -train_feat train.sparseX -train_target train.CT -dev_feat dev.sparseX -dev_target dev.CT

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.sparse import coo_matrix
import torch
import time
import threading

def stopwatch(start_time):
    while True:
        elapsed = time.time() - start_time
        print(f"\rElapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}", end="", flush=True)
        time.sleep(1)

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

# create dense array first to account for duplicate data entries (ie. 0 46 0.5 and 0 46 0.75)
# then create sparse array to remove zeros to save memory
def format_data(path, shape):
    dense_array = np.zeros(shape)

    with open(path, 'r') as file:
        for line in file:
            row, col, value = map(float, line.strip().split())
            dense_array[(int(row), int(col))] = value

    sparse_array = coo_matrix(dense_array)
    return sparse_array

def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = torch.nn.Linear(30, 8)
        self.output_linear = torch.nn.Linear(8, 1)

    def forward(self, x):
        a = torch.nn.functional.relu(self.hidden_linear(x))
        y_pred = self.output_linear(a)
        return y_pred
    
def initial_model(X_train, y_train, X_dev, y_dev):
    X = torch.Tensor(standardize(X_train))
    y = torch.Tensor(y_train)

    dev_X = torch.Tensor(standardize(X_dev))
    dev_y = torch.Tensor(y_dev)

    # train_model = torch.nn.Linear(train_components, 1)
    model = Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Instantiate optimizer object
    
    for i in range(5000):
        optimizer.zero_grad()

        y_pred = model(X)#.squeeze()
        print(y_pred)

        # Using torch's MSE
        # error = torch.nn.functional.mse_loss(y_pred, y)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        # Calculate gradients
        # error.backward()
        # optimizer.step() # Update parameters
        # optimizer.zero_grad() # Clear stored gradients
        print(f"\rUpdate: {i}\tLoss: {loss}", end="", flush=True)
    print()

def main():
    start = time.time()
    threading.Thread(target=stopwatch, args=(start,), daemon=True).start()
    
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
    # learnrate = args.learnrate

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
    y_train = np.loadtxt(traintarget_fn)

    # dev features
    devfeat_shape = (n_dev, d)
    devfeat_array = format_data(devfeat_fn, devfeat_shape)
    y_dev = np.loadtxt(devtarget_fn)

    print()
    print("arrays made", flush=True)

    # dimensionality reduction
    train_components = 30 # chosen based on figure
    train_tsvd = TruncatedSVD(n_components=train_components)
    X_train_tsvd = train_tsvd.fit(trainfeat_array).transform(trainfeat_array)
    X_train = train_tsvd.fit_transform(X_train_tsvd)

    dev_tsvd = TruncatedSVD(n_components=30)
    X_dev_tsvd = dev_tsvd.fit(devfeat_array).transform(devfeat_array)
    X_dev = dev_tsvd.fit_transform(X_dev_tsvd)

    print()
    print("reduced :)?", flush=True)

    # figure out how many components actually contribute/capture variance and adjust tsvd accordingly
    # explained = np.cumsum(dev_tsvd.explained_variance_ratio_)
    # plt.plot(explained)
    # plt.xlabel("Number of components")
    # plt.ylabel("Cumulative explained variance")
    # plt.title("Choosing n_components")
    # plt.grid(True)
    # plt.show()

    # model time 🥴
    print()
    print("it's modelin' time") # and model all over the place
    # initial_model(X_train, y_train, X_dev, y_dev)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print()
    print("fitted")

    y_pred_train = model.predict(X_train)
    print()
    print("train done")
    y_pred_dev = model.predict(X_dev)
    print()
    print("dev done")

    print()
    print('Logistic Regression train accuracy = %f' % accuracy_score(y_train, y_pred_train))
    print('Logistic Regression test accuracy = %f' % accuracy_score(y_dev, y_pred_dev))
    
    # precision ill-defined and being set to 0.0
    print()
    print('Logistic Regression train precision = %f' % precision_score(y_train, y_pred_train, average="macro"))
    print('Logistic Regression test precision = %f' % precision_score(y_dev, y_pred_dev, average="macro"))

    print()
    print('Logistic Regression train recall = %f' % recall_score(y_train, y_pred_train, average="macro"))
    print('Logistic Regression test recall = %f' % recall_score(y_dev, y_pred_dev, average="macro"))

    # from sklearn.svm import SVC

    # # Define Model.
    # svc_model = SVC(kernel='rbf', C=80)
    # print("made model")

    # # Train Model.
    # svc_model.fit(X_train, y_train)
    # print("fitted")

    # # Make Predictions with model.
    # y_pred_train = svc_model.predict(X_train)
    # print("predicted train")
    # y_dev_train = svc_model.predict(X_dev)
    # print("predicted dev")
    # print('SVC train accuracy = %f' % accuracy_score(y_train, y_pred_train))
    # print('SVC test accuracy = %f' % accuracy_score(y_dev, y_dev_train))
    

if __name__ == "__main__":
    main()

# wu-oh
# /home/simsj6/ThirdYear/SP25/471/FinalProject/task5/.venv/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. # when it's literally one iteration :)

# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
# Please also refer to the documentation for alternative solver options:
#     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#   n_iter_i = _check_optimize_result(
# fitted
# train done
# dev done
# Logistic Regression train accuracy = 0.117075
# Logistic Regression test accuracy = 0.079925
# /home/simsj6/ThirdYear/SP25/471/FinalProject/task5/.venv/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
# Logistic Regression train precision = 0.011452
# /home/simsj6/ThirdYear/SP25/471/FinalProject/task5/.venv/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
# Logistic Regression test precision = 0.006918
# Logistic Regression train recall = 0.018881
# Logistic Regression test recall = 0.011219
