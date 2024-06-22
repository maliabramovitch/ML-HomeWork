import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import map_feature
import plotDecisionBoundaryfunctions


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
    grad_J = (1 / m) * np.dot(X.T, (h_theta - y))
    grad_J[1:] += (lmbda / m) * theta[1:]
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


def plot_log_reg_line(X, y, theta, title='data'):
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
    x1 = X[:, 1]
    x2 = X[:, 2]
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    plt.plot(x1[y[:, 0] == 0], x2[y[:, 0] == 0], 'go',
             x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'rD',
             x1lh, x2lh, 'b-')
    plt.grid(axis='both'), plt.title(title), plt.show()


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
plt.title('Data scatter')
plt.show()

""" B """
onesvec = np.ones((m, 1))
X = np.concatenate((onesvec, X_orig), axis=1)
n = X.shape[1]
theta = np.zeros((n, 1))
y = y.reshape([y.shape[0], 1])
J, grad_J = computeCost(X, y, theta)
alpha = 0.2
num_iters = 10000
theta, J_iter = gradDescent_log(X, y, theta, alpha, num_iters)
plot_log_reg_line(X, y, theta, 'logistic linear regression')
print("The conclusion is that the model is too narrow for this kind of problem")

""" C """
X = map_feature.map_feature(x1, x2)

""" D, E """
n = X.shape[1]
theta = np.zeros((n, 1))
lmbda = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
thetas = []
for l in lmbda:
    theta, J_iter = gd_reg(X, y, theta, alpha, num_iters, l)
    thetas.append(theta)
    plotDecisionBoundaryfunctions.plotDecisionBoundary1(theta, X, y, 6, f'lambda = {l}')
print(
    "The effect of the lambda on the decision boundary is that as the lambda increases, so does the decision boundary.\n")

""" F """
Xdata = pd.read_csv("email_data_3_2024.csv")
data = Xdata.to_numpy()
X_orig = data[:, 0:2]
x1 = X_orig[:, 0]
x2 = X_orig[:, 1]
y = data[:, 2]
X = map_feature.map_feature(x1, x2)
y = y.reshape([y.shape[0], 1])
for i in range(6):
    correct_predictions = (np.sum(sigmoid(np.dot(X, thetas[i]))[y == 1] >= 0.5) +
                           np.sum(sigmoid(np.dot(X, thetas[i]))[y == 0] < 0.5))
    print(
        f"lambda={lmbda[i]}:\nThe number of correct samples that classified as ture: {correct_predictions}\nthe percentage of right identification is {correct_predictions / 100}\n");
