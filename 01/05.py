import numpy as np
import matplotlib.pyplot as plt


def data_normalization(X):
    mean1 = np.mean(X[:, 0])
    mean2 = np.mean(X[:, 1])
    div1 = np.std(X[:, 0])
    div2 = np.std(X[:, 1])
    X[:, 0] = (X[:, 0] - mean1) / div1
    X[:, 1] = (X[:, 1] - mean2) / div2
    return X, mean1, mean2, div1, div2


def set_data():
    data = np.loadtxt('houses.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2].reshape((-1, 1))
    return X, y


def show_data(X):
    data = np.loadtxt('faithful.txt')
    n_row = data.shape[0]
    x = data[:, 0].reshape(n_row, 1)
    y = data[:, 1].reshape(n_row, 1)
    return x, y


def cost_computation(X, y, q):
    m = X.shape[0]
    z = np.dot(X, q) - y
    J = 1 / (2 * m) * (np.dot(z.T, z)[0][0])
    return J


def gd_mini_batch(X, y, q, alpha, num_iter, mini_batch_size):
    m = X.shape[0]
    J_iter = np.zeros(num_iter)
    k = 0
    for j in range(num_iter):
        rand_frame = np.random.permutation(mini_batch_size)
        x_batch = 0
        y_batch = 0
        for i in rand_frame:
            start = i + mini_batch_size
            end = start + (m // mini_batch_size) + 1
            x_batch = X[start: end, :]
            y_batch = y[start: end, :]
            q -= alpha * (x_batch.T @ (np.dot(x_batch, q) - y_batch))
        J_iter[k] = cost_computation(x_batch, y_batch, q)
        k += 1
    return q, J_iter


def plot_reg_line_and_cost_3(X, y, theta, j_iter, iter):
    """
    plot_reg_line_and_cost plots the data points and regression line for linear regression
    along with the cost over iterations.

    Input arguments:
    X - np array (m, 3) - independent variables.
    y - np array (m, 1) - target variable.
    theta - np array (3, 1) - parameters.
    j_iter - list of cost values over iterations.
    iter - current iteration number.
    """
    fig = plt.figure(figsize=(14, 5))

    # 3D plot for regression data and line
    ax1 = fig.add_subplot(121, projection='3d')

    # Data points
    ax1.scatter(X[:, 1], X[:, 2], y, c='g', marker='o', label='Data Points')

    # Regression line
    x_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    y_vals = np.linspace(X[:, 2].min(), X[:, 2].max(), 100)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    z_vals = theta[0] + theta[1] * x_mesh.flatten() + theta[2] * y_mesh.flatten()

    # Plot the regression line
    ax1.plot(x_mesh.flatten(), y_mesh.flatten(), z_vals, 'r-', label='Regression Line')

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Target')
    ax1.set_title('3D Regression Data')
    ax1.legend()

    # Plot the cost over iterations
    ax2 = fig.add_subplot(122)
    ax2.plot(range(iter), j_iter[:iter])
    ax2.set_title('Cost over Iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.grid(True)

    fig.suptitle(f'Iteration: {iter}')
    fig.tight_layout()
    plt.show()


X, y = set_data()
show_data(X)
X, mean1, mean2, div1, div2 = data_normalization(X)
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
theta = np.zeros((3, 1))
alpha = 0.001
num_iter = 10000
q, J_iter = gd_mini_batch(X, y, theta, alpha, num_iter, mini_batch_size=16)
plot_reg_line_and_cost_3(X, y, theta, J_iter, num_iter)
print(f"theta0={theta[0]}    theta1={theta[1]}    theta2={theta[2]}")
h = theta[0] + theta[1] * 1200 + theta[2] * 5
print(f"The predicted cost for house with 1200 sf and 5 rooms: {h}")
