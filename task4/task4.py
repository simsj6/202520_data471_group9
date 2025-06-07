# Predicting the speech sound (“phoneme”) from an acoustic feature vector (describing the acoustic content in a small frame of audio). This is also a classification task.

# *********************************************
# get data
# *********************************************
# import wandb
import csv
# import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import xgboost
from xgboost import XGBClassifier
# import matplotlib as plot

# filepath strings:
paramsf = "proj_data/task4_phoneme/task4.config"
trainfeatsf = "proj_data/task4_phoneme/train.X"
traintargsf = "proj_data/task4_phoneme/train.CT"
devfeatsf = "proj_data/task4_phoneme/dev.X"
devtargsf = "proj_data/task4_phoneme/dev.CT"
testfeats = "proj_data/test.X"

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
csvfile = open("deliverables/task4.predictions", "w")
writer = csv.writer(csvfile)
dimensions = open(paramsf, "r").readlines()[3].split(" ")[-1].strip()
floatdim = float(dimensions)

print("starting tuning")
# sv_model_1 = LinearSVC(C=1.0)
# sv_model_2 = LinearSVC(C=0.1)
# sv_model_3 = LinearSVC(C=0.01)
# sv_model_4 = LinearSVC(C=2.0)
# sv_model_5 = LinearSVC(C=10.0)
# nn_model_6 = MLPClassifier()
# sv_model_7 = LinearSVC(C=10.0, dual=False)
# sv_model_8 = LinearSVC(C=1.0, dual=False)
# xg_model_9 = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=0.1, objective='multi:softmax')
# xg_model_10 = XGBClassifier(n_estimators=5, max_depth=4, learning_rate=0.01, objective='multi:softmax')
# xg_model_11 = XGBClassifier(n_estimators=20, max_depth=20, learning_rate=0.01, objective='multi:softmax')
# xg_model_12 = XGBClassifier(n_estimators=10, max_depth=10, learning_rate=0.1, objective='multi:softmax')
# xg_model_13 = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1, objective='multi:softmax')
xg_model_14 = XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.1, objective='multi:softmax')
# nn_model_15 = MLPClassifier(learning_rate_init=0.1, learning_rate='constant')
# nn_model_16 = MLPClassifier(learning_rate='adaptive')
# xg_model_17 = XGBClassifier(n_estimators=5000, max_depth=5, learning_rate=0.1, objective='multi:softmax')
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

# # tested
# writer.writerow(["model 5"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# sv_model_5 = sv_model_5.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = sv_model_5.predict(trainfeats)
# pred_dev = sv_model_5.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# tested
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

# # tested
# writer.writerow(["model 7"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# sv_model_7 = sv_model_7.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = sv_model_7.predict(trainfeats)
# pred_dev = sv_model_7.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# # tested
# writer.writerow(["model 8"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("data normalization")
# trainfeats_mean = trainfeats.mean()
# trainfeats_std = trainfeats.std()
# trainfeats_norm = (trainfeats - trainfeats_mean) / trainfeats_std
# devfeats_norm = (devfeats - trainfeats_mean) / trainfeats_std
# traintargs_mean = traintargs.mean()
# traintargs_std = traintargs.std()
# print("completed data normalization")
# print("beginning training")
# sv_model_8 = sv_model_8.fit(trainfeats_norm, traintargs)
# print("beginning predicting")
# pred_train = sv_model_8.predict(trainfeats_norm)
# pred_dev = sv_model_8.predict(devfeats_norm)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# # tested
# writer.writerow(["model 9"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("data normalization")
# trainfeats_mean = trainfeats.mean()
# trainfeats_std = trainfeats.std()
# trainfeats_norm = (trainfeats - trainfeats_mean) / trainfeats_std
# devfeats_norm = (devfeats - trainfeats_mean) / trainfeats_std
# traintargs_mean = traintargs.mean()
# traintargs_std = traintargs.std()
# print("completed data normalization")
# print("beginning training")
# xg_model_9 = xg_model_9.fit(trainfeats_norm, traintargs)
# print("beginning predicting")
# pred_train = xg_model_9.predict(trainfeats_norm)
# pred_dev = xg_model_9.predict(devfeats_norm)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# # tested
# writer.writerow(["model 10"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("data normalization")
# trainfeats_mean = trainfeats.mean()
# trainfeats_std = trainfeats.std()
# trainfeats_norm = (trainfeats - trainfeats_mean) / trainfeats_std
# devfeats_norm = (devfeats - trainfeats_mean) / trainfeats_std
# traintargs_mean = traintargs.mean()
# traintargs_std = traintargs.std()
# print("completed data normalization")
# print("beginning training")
# xg_model_10 = xg_model_10.fit(trainfeats_norm, traintargs)
# print("beginning predicting")
# pred_train = xg_model_10.predict(trainfeats_norm)
# pred_dev = xg_model_10.predict(devfeats_norm)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# # tested
# writer.writerow(["model 11"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# xg_model_11 = xg_model_11.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = xg_model_11.predict(trainfeats)
# pred_dev = xg_model_11.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# # tested
# writer.writerow(["model 12"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# xg_model_12 = xg_model_12.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = xg_model_12.predict(trainfeats)
# pred_dev = xg_model_12.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# tested
# writer.writerow(["model 13"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# xg_model_13 = xg_model_13.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = xg_model_13.predict(trainfeats)
# pred_dev = xg_model_13.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# tested 
writer.writerow(["model 14"])
writer.writerow(["training accuracy", "dev accuracy", "predictions"])
print("beginning training")
xg_model_14 = xg_model_14.fit(trainfeats, traintargs)
print("beginning predicting")
pred_train = xg_model_14.predict(trainfeats)
pred_dev = xg_model_14.predict(devfeats)
print("measuring loss")
writer.writerow([accuracy_score(traintargs, pred_train)])
writer.writerow([accuracy_score(devtargs, pred_dev)])
# writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
writer.writerow([pred_dev])
print("done")

# # tested
# writer.writerow(["model 15"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("data normalization")
# trainfeats_mean = trainfeats.mean()
# trainfeats_std = trainfeats.std()
# trainfeats_norm = (trainfeats - trainfeats_mean) / trainfeats_std
# devfeats_norm = (devfeats - trainfeats_mean) / trainfeats_std
# traintargs_mean = traintargs.mean()
# traintargs_std = traintargs.std()
# print("completed data normalization")
# print("beginning training")
# nn_model_15 = nn_model_15.fit(trainfeats_norm, traintargs)
# print("beginning predicting")
# pred_train = nn_model_15.predict(trainfeats_norm)
# pred_dev = nn_model_15.predict(devfeats_norm)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# # tested
# writer.writerow(["model 16"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# nn_model_16 = nn_model_16.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = nn_model_16.predict(trainfeats)
# pred_dev = nn_model_16.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# # tested 
# writer.writerow(["model 17"])
# writer.writerow(["training accuracy", "dev accuracy", "predictions"])
# print("beginning training")
# xg_model_17 = xg_model_17.fit(trainfeats, traintargs)
# print("beginning predicting")
# pred_train = xg_model_17.predict(trainfeats)
# pred_dev = xg_model_17.predict(devfeats)
# print("measuring loss")
# writer.writerow([accuracy_score(traintargs, pred_train)])
# writer.writerow([accuracy_score(devtargs, pred_dev)])
# # writer.writerow([f1_score(traintargs, pred_train, average='samples')])
# # writer.writerow([f1_score(devtargs, pred_dev, average='samples')])
# writer.writerow([pred_dev])
# print("done")

# final model used to predict:
print("beginning training")
xg_model_14 = xg_model_14.fit(trainfeats, traintargs)
print("beginning predicting")
predictions = xg_model_14.predict(testfeats)
writer.writerow([predictions])
print("done")