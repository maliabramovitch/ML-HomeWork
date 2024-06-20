import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
    """
    y = sigmoid(z) implements a sigmoid function for an input z
    """
    y = 1 / (1 + np.exp(-z))
    return y


def computeCost(X, y, theta):
    """
    COMPUTECOST Compute cost for logistic regression.
    Computes the cost of using theta parameters for logistic regression
    Input arguments: X - input matrix (np array), observations
    (features) in rows. y - (np array) output vector (labels) for
    each input sample, theta - (np array) parameters vector, weights of
    the measured features
    Output arguments:
    return - J - the cost function for theta
    Usage: J = computeCost(X, y, theta)
    """
    m = y.size
    J = []
    grad_J = np.zeros(theta.shape)
    z = np.dot(X, theta)
    h_theta = sigmoid(z)
    J = - (1 / m) * (np.dot(y.T, np.log(h_theta)) + np.dot((1 - y).T, np.log(1 - h_theta)))
    grad_J = (1 / m) * np.dot(X.T, (h_theta - y))
    return J, grad_J


def gradDescent_log(X, y, theta, alpha, num_iters):
    """
    gradDescent - Batch implementation of GD algorithm
    for logistic regression using matrix-vector operations
    Input arguments:
    X - (m, n) numpy matrix - each row is a feature vector,
    y - (m,1) np array of target values,
    theta - (n,1) initial parameter vector,
    alpha - learning rate
    num_iters - number of iterations.
    returns theta - parameters vector and J_iter - (num_iter,1)
    cost function vector.
    """
    J_iter = np.zeros(num_iters)
    for k in range(num_iters):
        J_iter[k], grad = computeCost(X, y, theta)
        theta = theta - alpha * grad
    return theta, J_iter


def computeCost_l(X, y, theta, lmbda):
    """
    COMPUTECOST Compute cost for logistic regression.
    Computes the cost of using theta parameters for logistic regression
    Input arguments: X - input matrix (np array), observations
    (features) in rows. y - (np array) output vector (labels) for
    each input sample, theta - (np array) parameters vector, weights of
    the measured features
    Output arguments:
    return - J - the cost function for theta
    Usage: J = computeCost(X, y, theta)
    """
    m = y.size
    J = []
    grad_J = np.zeros(theta.shape)
    z = np.dot(X, theta)
    h_theta = sigmoid(z)
    J = - (1 / m) * (np.dot(y.T, np.log(h_theta)) + np.dot((1 - y).T, np.log(1 - h_theta)))
    J += (lmbda / 2 * m) * np.dot(theta.T, theta)
    x_j0 = X[:, 0].reshape(m, 1)
    x_jrest = X[:, 1:].reshape(m, X.shape[1] - 1)
    grad_J[0] = (1 / m) * np.dot((h_theta * x_j0 - y).T, x_j0)
    grad_J[1:] = (1 / m) * np.dot((np.dot(h_theta.T, x_jrest) - y).T, x_jrest) + (lmbda / m)
    return J, grad_J


def gd_reg(X, y, theta, alpha, num_iters, lmbda):
    """
    gradDescent - Batch implementation of GD algorithm
    for logistic regression using matrix-vector operations
    Input arguments:
    X - (m, n) numpy matrix - each row is a feature vector,
    y - (m,1) np array of target values,
    theta - (n,1) initial parameter vector,
    alpha - learning rate
    num_iters - number of iterations.
    returns theta - parameters vector and J_iter - (num_iter,1)
    cost function vector.
    """
    J_iter = np.zeros(num_iters)
    for k in range(num_iters):
        J_iter[k], grad = computeCost_l(X, y, theta, lmbda)
        theta = theta - alpha * grad
    return theta, J_iter


def map_feature(x1, x2, degree=6):
    '''
    Maps a two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    x1, x2, x1 ** 2, x2 ** 2, x1*x2, x1*x2 ** 2, etc...
    The inputs x1, x2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    X = np.ones(shape=(x1[:, 0].size, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            X = np.append(X, r, axis=1)

    return X


""" A """
Xdata = pd.read_csv("email_data_2.csv")
data = Xdata.to_numpy()
X_orig = data[:, 0:2]
y = data[:, 2]
m = y.size
x1 = X_orig[:, 0]
x2 = X_orig[:, 1]
plt.figure(1)
plt.plot(X_orig[y == 0, 0], X_orig[y == 0, 1], 'go', X_orig[y == 1, 0], X_orig[y == 1, 1], 'rD'),
plt.grid(axis='both')
plt.show()

""" B """
onesvec = np.ones((m, 1))
X = np.concatenate((onesvec, X_orig), axis=1)
n = X.shape[1]
theta = np.zeros((n, 1))
y = y.reshape([y.shape[0], 1])
J, grad_J = computeCost(X, y, theta)
alpha = 0.001
num_iters = 90000
theta, J_iter = gradDescent_log(X, y, theta, alpha, num_iters)
print("The conclusion is that the model is too narrow for this kind of problem")

""" C """
X = map_feature(x1, x2)

""" D """
n = X.shape[1]
theta = np.zeros((n, 1))
theta, J_iter = gd_reg(X, y, theta, alpha, num_iters, lmbda=0)
