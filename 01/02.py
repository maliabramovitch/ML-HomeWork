import numpy as np
import matplotlib.pyplot as plt


# """def plot_reg_line_and_cost(X, y, theta, j_iter, iter):
#     """
#     plot_reg_line plots the data points and regression line
#     for linear regrssion
#     Input arguments: X - np array (m, n) - independent variable.
#     y - np array (m,1) - target variable
#     theta - parameters
#     """
#     if X.shape[1] == 2:
#         ind = 1
#     else:
#         ind = 0
#
#     x_min = X[:, ind].min()
#     x_max = X[:, ind].max()
#     ind_min = X[:, ind].argmin()
#     ind_max = X[:, ind].argmax()
#     y_min = y[ind_min]
#     y_max = y[ind_max]
#     Xlh = X[(ind_min, ind_max), :]
#     yprd_lh = np.dot(Xlh, theta)
#     fig, axe = plt.subplots(1, 2)
#     axe[0].plot(X[:, ind], y, 'go', Xlh, yprd_lh, 'm-')
#     axe[0].axis((x_min - 5, x_max + 5, min(y_min, y_max) - 5, max(y_min, y_max) + 5))
#     # axe[0].xlabel('x'), plt.ylabel('y'),
#     axe[0].set_title('Regression data')
#     axe[0].grid()
#
#     axe[1].plot(j_iter[:iter])
#     axe[1].set_title('Cost')
#     axe[1].grid()
#
#     fig.suptitle(f'iter: {iter}')
#     fig.tight_layout()
#     fig.show()
#
#
# def compute_cost(X, y, theta):
#     m = y.shape[0]
#     z = np.dot(X, theta) - y
#     J = (1 / 2 * m) * np.dot(z.T, z)
#     # J = (1 / 2 * m) * np.sum(z ** 2)
#     return J
#
#
# def gd_ol(X, y, theta, alpha, num_iter):
#     m = y.shape[0]
#     J_iter = np.zeros((num_iter * m))
#     k = 0
#     for j in range(num_iter):
#         randindex = np.random.permutation(m)
#         for i in range(m):
#             xi = X[randindex[i], :]
#             xi = xi.reshape(1, xi.shape[0])
#             yi = y[randindex[i]]
#             delta = np.dot(xi, theta) - yi
#             theta = theta - alpha * delta * xi.T
#             J = compute_cost(X, y, theta)[0][0]
#             J_iter[k] = J
#             k += 1
#     return theta, J_iter
#
#
# data = np.load('Cricket.npz')
# sorted(data)
# yx = data['arr_0']
# x = yx[:, 1]
# y = yx[:, 0]
# x = x.reshape(x.shape[0], 1)
# y = y.reshape(y.shape[0], 1)
#
# m = y.size
# onesvec = np.ones((m, 1))
# X = np.concatenate((onesvec, x), axis=1)
# alpha = 0.0001
# num_iter = 100
# n1 = X.shape[1]
# theta = np.zeros((n1, 1))
# theta, J_iter = gd_ol(X, y, theta, alpha, num_iter)
#
# plot_reg_line_and_cost(X, y, theta, J_iter, num_iter)"""


# showing the data
def show_data():
    data = np.load('Cricket.npz')
    yx = data['arr_0']
    x = yx[:, 1]
    y = yx[:, 0]

    plt.plot(x, y, 'or')
    plt.grid()
    plt.show()
    return x, y, plt


def cost_computation(X, y, q):
    m = y.shape[0]
    z = np.dot(X, q) - y
    J = (1 / 2 * m) * np.dot(z.T, z)
    return J

def gd_batch(X, y, q, alpha, num_iter):
    m = y.shape[0]
    J_iter = np.zeros((num_iter * m))
    k = 0
    for j in range(num_iter):
        randindex = np.random.permutation(m)
        for i in range(m):
            xi = X[randindex[i], :]
            xi = xi.reshape(1, xi.shape[0])
            yi = y[randindex[i]]
            delta = np.dot(xi, theta) - yi
            theta = theta - alpha * delta * xi.T
            J = cost_computation(X, y, theta)[0][0]
            J_iter[k] = J
            k += 1
    return theta, J_iter


show_data()
