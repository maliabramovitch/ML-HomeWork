# College admittance decision using logistic regression
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
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


def plot_log_reg_line(X, y, theta):
    """
    plot_reg_line plots the data points and regression line for logistic regrssion
    Input arguments: X - np array (m, n) - independent variable.
    y - np array (m,1) - target variable
    theta - parameters
    The function is for 2-d input- x2 = -(theta[0] + theta[1]*x1)/theta[2]
    """
    ind = 1
    x1_min = 0.9 * X[:, ind].min()
    x1_max = 1.1 * X[:, ind].max()
    x2_min = - (theta[0] + theta[1] * x1_min) / theta[2]
    x2_max = - (theta[0] + theta[1] * x1_max) / theta[2]
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    x1 = X[:, 1]
    x2 = X[:, 2]
    plt.plot(x1[y[:, 0] == 0], x2[y[:, 0] == 0], 'ro',
             x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'go',
             x1lh, x2lh, 'b-')
    plt.xlabel('Exam 1'), plt.ylabel('Exam 2'), plt.title('data'),
    plt.grid(axis='both'), plt.show()


""" A """
Xdata = pd.read_csv("admittance_data.csv")
data = Xdata.to_numpy()
X_orig = data[:, 0:2]
y = data[:, 2]
m = y.size

""" B """
x1 = X_orig[:, 0]
x2 = X_orig[:, 1]
plt.figure(1)
plt.plot(X_orig[y == 0, 0], X_orig[y == 0, 1], 'ro', X_orig[y == 1, 0], X_orig[y == 1, 1], 'go'),
plt.grid(axis='both'),
plt.xlabel("test no. 1")
plt.ylabel("test no. 2")
plt.show()

""" C """
# functions

""" D """
onesvec = np.ones((m, 1))
X = np.concatenate((onesvec, X_orig), axis=1)
n = X.shape[1]
theta = np.zeros((n, 1))
y = y.reshape([y.shape[0], 1])
J, grad_J = computeCost(X, y, theta)
alpha = 0.001
num_iters = 90000
theta, J_iter = gradDescent_log(X, y, theta, alpha, num_iters)
plt.plot(J_iter)
plt.show()
plot_log_reg_line(X, y, theta)

""" E F G """
x1_mean = np.mean(x1)
x2_mean = np.mean(x2)
x1_std = np.std(x1)
x2_std = np.std(x2)
X_norm = X
X_norm[:, 1] = (X_norm[:, 1] - x1_mean) / x1_std
X_norm[:, 2] = (X_norm[:, 2] - x2_mean) / x2_std
theta = np.zeros((n, 1))
y = y.reshape([y.shape[0], 1])
J, grad_J = computeCost(X_norm, y, theta)
alpha = 0.02
num_iters = 3000
theta, J_iter = gradDescent_log(X_norm, y, theta, alpha, num_iters)
plt.plot(J_iter)
plt.title("Cost function with normalization")
plt.show()
plot_log_reg_line(X_norm, y, theta)

""" H """
z = lambda x1, x2: theta[0] + theta[1] * ((x1-x1_mean)/x1_std) + theta[2] * ((x2-x2_mean)/x2_std)
h1 = sigmoid(z(65,42))
print(f"The probability of students with grades [65,42] is: {h1}")
h2 = sigmoid(z(53,85))
print(f"The probability of students with grades [53,85] is: {h2}")
