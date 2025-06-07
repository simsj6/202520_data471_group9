# Name: Jenny Sims
# Date: 06/05/25
# Description: Predicting the beer style from the specific ingredients (e.g., hops, yeasts, or fermentables) used to brew the beer. This is a classification task.

# example call:
# python task5.py -config task5.config -train_feat train.sparseX -train_target train.CT -dev_feat dev.sparseX -dev_target dev.CT -test test.sparseX

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
from xgboost import XGBClassifier

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
    parser.add_argument('-test', type=str, required=True,
                        help='Test set target file')
    
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
    test_fn = args.test

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
    X_train = format_data(trainfeat_fn, trainfeat_shape)
    y_train = np.loadtxt(traintarget_fn)

    # dev features
    devfeat_shape = (n_dev, d)
    X_dev = format_data(devfeat_fn, devfeat_shape)
    y_dev = np.loadtxt(devtarget_fn)

    print("\narrays made", flush=True)


    # BASELINE CALCULATION
    # train_classes = [0] * c
    # for i in range(n_train):
    #     train_classes[int(y_train[i])] += 1
    
    # train_majority = max(train_classes)
    # train_baseline = train_majority / n_train

    # dev_classes = [0] * c
    # for i in range(n_dev):
    #     dev_classes[int(y_dev[i])] += 1
    
    # dev_majority = max(dev_classes)
    # dev_baseline = dev_majority / n_dev

    # print("Train baseline: %f" % train_baseline)
    # print("Dev baseline: %f" % dev_baseline)


    # DIMENSIONALITY REDUCTION
    # train_components = 30 # chosen based on figure
    # train_tsvd = TruncatedSVD(n_components=train_components)
    # X_train = train_tsvd.fit(trainfeat_array).transform(trainfeat_array)

    # dev_tsvd = TruncatedSVD(n_components=30)
    # X_dev_tsvd = dev_tsvd.fit(devfeat_array).transform(devfeat_array)
    # X_dev = train_tsvd.transform(devfeat_array)

    # print("\nreduced :)?", flush=True)

    # figure out how many components actually contribute/capture variance and adjust tsvd accordingly
    # explained = np.cumsum(dev_tsvd.explained_variance_ratio_)
    # plt.plot(explained)
    # plt.xlabel("Number of components")
    # plt.ylabel("Cumulative explained variance")
    # plt.title("Choosing n_components")
    # plt.grid(True)
    # plt.show()

    # model time 🥴
    print("\nit's modelin' time") # and model all over the place

    # XGBOOST
    # change to 1000 and 50
    n_est = 1000
    max_depth = 50
    lr = 0.1
    objective = "multi:softmax"
    print("\nXGBoost with n_estimators=%d, max_depth=%d, learning_rate=%.3f, objective=%s" % (n_est, max_depth, lr, objective))
    bst = XGBClassifier(n_estimators=n_est, max_depth=max_depth, learning_rate=lr, objective=objective)
    # fit model
    bst.fit(X_train, y_train)
    print("\nfitted")
    # make predictions
    y_pred_train = bst.predict(X_train)
    y_dev_train = bst.predict(X_dev)

    print('\nXGBoost train accuracy = %f' % accuracy_score(y_train, y_pred_train))
    print('XGBoost dev accuracy = %f' % accuracy_score(y_dev, y_dev_train))


    # TEST SET OUTPUT
    test_array = np.loadtxt(test_fn)
    rows = int(max(test_array[:,0])) + 1    # add one to account for 0 indexing
    # cols = int(max(test_array[:,1])) + 1  # NEEDS TO BE SAME AS D
    cols = d
    X_test = format_data(test_fn, (rows, cols))
    y_pred_test = bst.predict(X_test)
    np.savetxt("task5.predictions", y_pred_test, fmt="%d")
    print("\nAll done! :)")


    # NEURAL NETWORK
    # initial_model(X_train, y_train, X_dev, y_dev)


    # time.sleep(1)
    # # LOGISTIC REGRESSION
    # max_iterations = 1000
    # print("\nLogistic Regression with max_iter=%d" % (max_iterations))
    # model = LogisticRegression(max_iter=max_iterations)
    # model.fit(X_train, y_train)
    # # print("\nfitted")

    # y_pred_train = model.predict(X_train)
    # # print("\ntrain done")
    # y_pred_dev = model.predict(X_dev)
    # print("\npredicting done")

    # print('Logistic Regression train accuracy = %f' % accuracy_score(y_train, y_pred_train))
    # print('Logistic Regression test accuracy = %f' % accuracy_score(y_dev, y_pred_dev))
    
    # # precision ill-defined and being set to 0.0
    # print('\nLogistic Regression train precision = %f' % precision_score(y_train, y_pred_train, average="macro"))
    # print('Logistic Regression test precision = %f' % precision_score(y_dev, y_pred_dev, average="macro"))

    # print('\nLogistic Regression train recall = %f' % recall_score(y_train, y_pred_train, average="macro"))
    # print('Logistic Regression test recall = %f' % recall_score(y_dev, y_pred_dev, average="macro"))


    # SVM
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

