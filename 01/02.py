import numpy as np
import matplotlib.pyplot as plt


def plot_reg_line_and_cost(X, y, theta, j_iter, iter, alpha):
    """
    plot_reg_line plots the data points and regression line
    for linear regrssion
    Input arguments: X - np array (m, n) - independent variable.
    y - np array (m,1) - target variable
    theta - parameters
    """
    if X.shape[1] == 2:
        ind = 1
    else:
        ind = 0

    x_min = X[:, ind].min()
    x_max = X[:, ind].max()
    ind_min = X[:, ind].argmin()
    ind_max = X[:, ind].argmax()
    y_min = y[ind_min]
    y_max = y[ind_max]
    Xlh = X[(ind_min, ind_max), :]
    yprd_lh = np.dot(Xlh, theta)
    (fig, axe) = plt.subplots(1, 2)
    axe[0].plot(X[:, ind], y, 'mo', Xlh, yprd_lh, 'c-')
    axe[0].axis((x_min - 3, x_max + 3, min(y_min, y_max) - 3, max(y_min, y_max) + 3))
    axe[0].set_title('Regression data')
    axe[0].grid()

    axe[1].plot(j_iter[:iter], 'r')
    axe[1].set_title('Cost')
    axe[1].grid()
    alpha_str = "{:.5f}".format(alpha)
    fig.suptitle(f'iter: {iter}    Alpha: {alpha_str}')
    plt.tight_layout()
    fig.show()


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
    J_iter = np.zeros((num_iter))
    k = 0
    for j in range(num_iter):
        randindex = np.random.permutation(m)
        for i in range(m):
            xi = X[randindex[i], :]
            xi = xi.reshape(1, xi.shape[0])
            yi = y[randindex[i]]
            delta = np.dot(xi, q) - yi
            q -= alpha * delta * xi.T
        J = cost_computation(X, y, q)[0][0]
        J_iter[k] = J
        k += 1
    return q, J_iter


show_data()
data = np.load('Cricket.npz')
sorted(data)
yx = data['arr_0']
x = yx[:, 1]
y = yx[:, 0]
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

m = y.size
onesvec = np.ones((m, 1))
X = np.concatenate((onesvec, x), axis=1)
alpha = 0.00002
num_iter = 100
n1 = X.shape[1]
theta = np.zeros((n1, 1))
theta, J_iter = gd_batch(X, y, theta, alpha, num_iter)
print(f"theta0 = {theta[0]}    theta1 = {theta[1]}")
plot_reg_line_and_cost(X, y, theta, J_iter, num_iter, alpha)
