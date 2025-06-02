import torch
import sklearn as sk
import numpy as np

class dataset:
    def __init__(this):
        this.file = "./task3_ratings/dev.sparseX"
        this.fileConfig = "./task3_ratings/task3.config"
        this.features = np.loadtxt(this.file)
        this.data = this.load_data()

    def parser(this):
        row = this.features[:, 0]
        col = this.features[:, 1]
        val = this.features[:, 2]
        return row, col, val

    # Load config values
    def load_config(this, path):
        with open(path, 'r') as f:
            token = f.read().split()
        n_train = int(token[1])
        n_dev = int(token[3])
        d = int(token[5])
        c = int(token[7])
        return n_train, n_dev, d, c
    
    def load_data(this):
        n_train, n_dev, d, c = this.load_config(this.fileConfig)
        data = np.zeros((n_dev, d))
        for i in this.features:
            row, col, val = this.parser()
            data[row, col] = val
            
        return data
    
print("Gathering data")
obj = dataset()
print(obj.data)
print("done")