# Predicting the speech sound (“phoneme”) from an acoustic feature vector (describing the acoustic content in a small frame of audio). This is also a classification task.

# *********************************************
# get data
# *********************************************

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC

# filepath strings:
paramsf = "proj_data/task4_phoneme/task4. config"
trainfeatsf = "proj_data/task4_phoneme/train.X"
traintargsf = "proj_data/task4_phoneme/train. CT"
devfeatsf = "proj_data/task4_phoneme/dev.X"
devtargsf = "proj_data/task4_phoneme/dev.CT"

trainfeats = np.loadtxt(trainfeatsf)
traintargs = np.loadtxt(traintargsf)
devfeats = np.loadtxt(devfeatsf)
devtargs = np.loadtxt(devtargsf)

# *********************************************
# put into usable format
# *********************************************


# *********************************************
# set up model
# *********************************************
dimensions = open(paramsf, "r").readlines()[3].split(" ")[-1].strip()
floatdim = float(dimensions)
sv_model = LinearSVC()

# *********************************************
# train?
# *********************************************
print("beginning training")
sv_model.fit(trainfeats, traintargs)
print("finished training")

# *********************************************
# make predictions
# *********************************************
print("beginning predictions")
pred_train = sv_model.predict(trainfeats)
pred_dev = sv_model.predict(devfeats)
print("finished predictions")

# *********************************************
# measure accuracy
# *********************************************
print("SVC train mse = %f" % mean_squared_error(traintargs, pred_train))
print("SVC dev mse = %f" % mean_squared_error(devtargs, pred_dev))
