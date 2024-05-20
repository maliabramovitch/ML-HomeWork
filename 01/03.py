import numpy as np
import matplotlib.pyplot as plt


def plot_reg_line_and_cost(X, y, theta, j_iter, iter, alpha):
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
    fig.suptitle(f'Iter: {iter}    Alpha: {alpha_str}')
    plt.tight_layout()
    fig.show()


def show_data():
    data = np.loadtxt('faithful.txt')
    n_row = data.shape[0]
    x = data[:, 0].reshape(n_row, 1)
    y = data[:, 1].reshape(n_row, 1)
    plt.plot(x, y, 'xr')
    plt.title('Faithful Geyser')
    plt.xlabel('Duration of minutes of the eruption')
    plt.ylabel('Time to next eruption (minutes)')
    plt.show()
    return x, y


def cost_computation(X, y, q):
    m = y.shape[0]
    z = np.dot(X, q) - y
    J = 1 / (2 * m) * (np.dot(z.T, z)[0][0])
    return J


def gd_mini_batch(X, y, q, alpha, num_iter, mini_batch_size):
    m = y.shape[0]
    J_iter = np.zeros(num_iter)
    k = 0
    for j in range(num_iter):
        rand_frame = np.random.permutation(mini_batch_size)
        for i in rand_frame:
            start = i * mini_batch_size
            end = start + (m // mini_batch_size) + 1
            x_batch = X[start: end, :]
            y_batch = y[start: end, :]
            q -= alpha * (x_batch.T @ (np.dot(x_batch, q) - y_batch))
        J_iter[k] = cost_computation(x_batch, y_batch, q)
        k += 1
    return q, J_iter


x, y = show_data()
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)
m = x.shape[0]
X = np.concatenate((np.ones((m, 1)), x), axis=1)
theta = np.random.random((2, 1))
alpha = 0.001
num_iter = 2000
mini_batch_size = 16
theta, J_iter = gd_mini_batch(X, y, theta, alpha, num_iter, mini_batch_size)
print(f'theta0 = {theta[0]}    theta1 = {theta[1]}')
print(f'Cost = {J_iter[-1]}')
for m in [2.1, 3.5, 5.2]:
    h = theta[0] + theta[1] * m
    print(f'the prediction for explosion with {m} duration = {h}')
plot_reg_line_and_cost(X, y, theta, J_iter, num_iter, alpha)

# the best alpha values are between: 0.001 - 0.000005
