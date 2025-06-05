# Predicting the speech sound (“phoneme”) from an acoustic feature vector (describing the acoustic content in a small frame of audio). This is also a classification task.

# *********************************************
# get data
# *********************************************
import wandb
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib as plot

# filepath strings:
paramsf = "proj_data/task4_phoneme/task4.config"
trainfeatsf = "proj_data/task4_phoneme/train.X"
traintargsf = "proj_data/task4_phoneme/train.CT"
devfeatsf = "proj_data/task4_phoneme/dev.X"
devtargsf = "proj_data/task4_phoneme/dev.CT"

trainfeats = np.loadtxt(trainfeatsf)
traintargs = np.loadtxt(traintargsf)
devfeats = np.loadtxt(devfeatsf)
devtargs = np.loadtxt(devtargsf)

# *********************************************
# put into usable format
# *********************************************
# thought about using sparisfy but it says it can actually be less efficient for data that is less than 50% zeroes so i did not

# *********************************************
# set up model
# *********************************************
csvfile = open("task4/forplot5.csv", "w")
writer = csv.writer(csvfile)
dimensions = open(paramsf, "r").readlines()[3].split(" ")[-1].strip()
floatdim = float(dimensions)

print("starting tuning")
# sv_model_1 = LinearSVC(C=1.0)
# sv_model_2 = LinearSVC(C=0.1)
# sv_model_3 = LinearSVC(C=0.01)
# sv_model_4 = LinearSVC(C=2.0)
sv_model_5 = LinearSVC(C=10.0)
# nn_model_6 = MLPClassifier()
print("finished tuning")

# *********************************************
# train each model, predict, then track results
# *********************************************

# tested
# writer.writerow(["model 1"])
# writer.writerow(["training accuracy", "dev accuracy", "precision-recall training", "precision-recall dev", "f1 training", "f1 dev", "predictions"])
# print("beginning training")
# sv_model_1 = sv_model_1.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = sv_model_1.predict(trainfeats)
# pred_dev = sv_model_1.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# tested
# writer.writerow(["model 2"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# sv_model_2 = sv_model_2.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = sv_model_2.predict(trainfeats)
# pred_dev = sv_model_2.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# tested
# writer.writerow(["model 3"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# sv_model_3 = sv_model_3.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = sv_model_3.predict(trainfeats)
# pred_dev = sv_model_3.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# tested
# writer.writerow(["model 4"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# sv_model_4 = sv_model_4.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = sv_model_4.predict(trainfeats)
# pred_dev = sv_model_4.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# testing
writer.writerow(["model 5"])
writer.writerow(["training accuracy", "dev accuracy", "predictions"])
print("beginning training")
sv_model_5 = sv_model_5.fit(trainfeats, traintargs)
print("beginning predicting")
pred_train = sv_model_5.predict(trainfeats)
pred_dev = sv_model_5.predict(devfeats)
print("measuring loss")
writer.writerow([accuracy_score(traintargs, pred_train)])
writer.writerow([accuracy_score(devtargs, pred_dev)])
# writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
writer.writerow([pred_dev])
print("done")

# # untested
# writer.writerow(["model 6"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# nn_model_6 = nn_model_6.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = nn_model_6.predict(trainfeats)
# pred_dev = nn_model_6.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")