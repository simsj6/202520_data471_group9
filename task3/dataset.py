import torch
import sklearn as sk
import numpy as np

class dataset:
    def __init__(this, file):
        this.file = "./task3_ratings/dev.sparseX"
        this.features = np.loadtxt(file)
        this.data = np.(())

    def parser(this, i):
        row = this.features[:, 0]
        col = this.features[:, 1]
        val = this.features[:, 2]


    