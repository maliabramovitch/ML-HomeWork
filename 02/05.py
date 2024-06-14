import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" A """


def plot_points_and_boundary(X, y, theta, b=0):
    ind = 1
    x1_min = 1.1 * X[:, ind].min()
    x1_max = 1.1 * X[:, ind].max()
    x2_min = - (b + theta[0] * x1_min) / theta[1]
    x2_max =  (b + theta[0] * x1_max) / theta[1]
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    x1 = X[:, 0]
    x2 = X[:, 1]
    plt.plot(x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'go',
             x1[y[:, 0] == -1], x2[y[:, 0] == -1], 'rx',
             x1lh, x2lh, 'b-')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('data')
    plt.grid(axis='both')
    plt.show()


def perceptron_train(X, y, plotflag, max_iter):
    num_correct = 0
    mat_shape = X.shape
    if len(mat_shape) > 1:
        nrow = mat_shape[0]
        ncol = mat_shape[1]
    else:
        X = X.reshape(X.shape[0], 1)

    current_index = 0
    theta = np.zeros((ncol, 1))
    b = 0
    j = 0
    k = 0
    is_first_iter = 1
    while num_correct < nrow and k < max_iter:
        j = j + 1
        xt = X[current_index, :]
        xt = xt.reshape(xt.shape[0], 1)
        yt = y[current_index]
        # ---------------------------------------------------------------------
        a =  # your code here (one line). if the sign of the hypothesis function
        # is not equal yt it should be negative, otherwise it should be
        # positive. Include here the bias term b in your code and assign
        # b = 0.
        # ---------------------------------------------------------------------
        if is_first_iter == 1 or a < 0:
            # -----------------------------------------------------------------
            ####### your code here (3 lines)
            theta =
            num_correct =  # it should be zeroed after each error
            k =  # this counts the iterations (i.e. the number of error occurences)
            #########
            # -----------------------------------------------------------------
            is_first_iter = 0
            if plotflag == 1:
                plot_points_and_boundary(X, y, theta, b)
                plt.pause(0.01)
        else:
            num_correct += 1
            # print(num_correct)
        current_index += 1
        if current_index > nrow - 1:
            current_index = 1
    return theta, k


npzfile = np.load("Perceptron_exercise_2.npz")
sorted(npzfile.files)
X = npzfile['arr_0']
y = npzfile['arr_1']
