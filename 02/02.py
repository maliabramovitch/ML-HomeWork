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


def plot_log_reg_line(X, y, theta, quad=False):
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
    if quad:
        x1 = np.linspace(x1_min, x1_max, 1000)
        x2 = - ((theta[3] * (x1 ** 2) + theta[1] * x1 + theta[0]) * (1 / theta[2]))
        plt.plot(X[y[:, 0] == 0, ind], X[y[:, 0] == 0, ind + 1], 'go',
                 X[y[:, 0] == 1, ind], X[y[:, 0] == 1, ind + 1], 'r^', x1, x2, 'b-')
    else:
        x1 = X[:, 1]
        x2 = X[:, 2]
        x1lh = np.array([x1_min, x1_max])
        x2lh = np.array([x2_min, x2_max])
        plt.plot(x1[y[:, 0] == 0], x2[y[:, 0] == 0], 'go',
                 x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'r^',
                 x1lh, x2lh, 'b-')
    plt.xlabel('feature no. 1'), plt.ylabel('feature no. 2'), plt.title('data'),
    plt.grid(axis='both'), plt.show()


""" A """
Xdata = pd.read_csv("email_data_1.csv")
data = Xdata.to_numpy()
X_orig = data[:, 0:2]
y = data[:, 2]
m = y.size
x1 = X_orig[:, 0]
x2 = X_orig[:, 1]
plt.figure(1)
plt.plot(X_orig[y == 0, 0], X_orig[y == 0, 1], 'go', X_orig[y == 1, 0], X_orig[y == 1, 1], 'r^'),
plt.grid(axis='both'),
plt.xlabel("feature no. 1")
plt.ylabel("feature no. 2")
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
plot_log_reg_line(X, y, theta)

""" C """
x1_vec = x1.reshape((m, 1))
X_quad = np.concatenate((X, x1_vec ** 2), axis=1)
n = X_quad.shape[1]
theta = np.zeros((n, 1))
y = y.reshape([y.shape[0], 1])
J, grad_J = computeCost(X_quad, y, theta)
alpha = 0.2
num_iters = 50000
theta, J_iter = gradDescent_log(X_quad, y, theta, alpha, num_iters)
plot_log_reg_line(X_quad, y, theta, True)

""" D """
plt.plot(J_iter)
plt.title("Cost function")
plt.show()

""" E """
z_lin = lambda x1, x2: theta[0] + theta[1] * x1 + theta[2] * x2
z_quad = lambda x1, x2, x3: theta[0] + theta[1] * x1 + theta[2] * x2 + theta[3] * x3
Xdata = pd.read_csv("email_data_test_2024.csv")
data = Xdata.to_numpy()
X_orig = data[:, 0:2]
x1_vec = X_orig[:, 0].reshape((X_orig.shape[0], 1))
X_quad = np.concatenate((X_orig, x1_vec ** 2), axis=1)
y = data[:, 2]
index = 0
correct_predictions = 0
for x1, x2 in X_orig:
    h = sigmoid(z_lin(x1, x2))[0]
    correct_predictions += 1 if ((h >= 0.5 and y[index] == 1) or (h < 0.5 and y[index] == 0)) else 0
    index += 1
print(
    f"Linear:\nThe number of correct samples that classified as ture is: {correct_predictions}\nthe percentage of right identification is {correct_predictions / X_orig.shape[0]}")
print()
index = 0
correct_predictions = 0
for x1, x2, x3 in X_quad:
    h = sigmoid(z_quad(x1, x2, x3))[0]
    correct_predictions += 1 if ((h > 0.5 and y[index] == 1) or (h < 0.5 and y[index] == 0)) else 0
    index += 1
print(
    f"Quad:\nThe number of correct samples that classified as ture is: {correct_predictions}\nthe percentage of right identification is {correct_predictions / X_orig.shape[0]}")
