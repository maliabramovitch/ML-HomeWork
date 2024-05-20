import numpy as np
import matplotlib.pyplot as plt


def data_normalization(X):
    mean1 = np.mean(X[:, 0])
    mean2 = np.mean(X[:, 1])
    mean3 = np.mean(X[:, 2])
    div1 = np.std(X[:, 0])
    div2 = np.std(X[:, 1])
    div3 = np.std(X[:, 2])
    X[:, 0] = (X[:, 0]-mean1)/div1
    X[:, 1] = (X[:, 1]-mean2)/div2
    X[:, 2] = (X[:, 2]-mean3)/div3
    return X, mean1, mean2, mean3, div1, div2, div3




X = np.loadtxt('houses.txt', delimiter=',')
data_normalization(X)
