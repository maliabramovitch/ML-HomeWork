import numpy as np
import matplotlib.pyplot as plt


def plot_reg_line_and_cost(X, y, theta, j_iter, iter, alpha):
    '''

    '''
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
    axe[0].plot(X[:, ind], y, 'm.', Xlh[:, 1], yprd_lh, 'c-')
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
    data = np.genfromtxt(fname='kleibers_law_data.csv', skip_header=1, delimiter=',')
    X = np.log(data[:, 0])
    X = X.reshape(-1, 1)
    y = np.log(data[:, 1])
    y = y.reshape((-1, 1))
    plt.plot(X, y, 'm.')
    plt.title('Cost of house in 100K ILS')
    plt.xlabel("House's front Length in meters")
    plt.ylabel("House's cost in 100K ILS")
    plt.show()
    return X, y


def cost_computation(X, y, q):
    m = X.shape[0]
    z = np.dot(X, q) - y
    J = 1 / (2 * m) * (np.dot(z.T, z)[0][0])
    return J


def gd_mini_batch(X, y, theta, q, num_iter, batch_size):
    m = X.shape[0]
    J_iter = np.zeros(num_iter)
    for j in range(num_iter):
        indices = np.arange(m)
        np.random.shuffle(indices)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            delta = np.dot(X_batch, theta) - y_batch
            theta = theta - q * np.dot(X_batch.T, delta)

        J = cost_computation(X, y, theta)
        J_iter[j] = J

    return theta, J_iter


# A
X, y = show_data()
# B
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
theta = np.zeros((2, 1))
alpha = 0.001
num_iter = 200
q, J_iter = gd_mini_batch(X, y, theta, alpha, num_iter, 14)
# C
plot_reg_line_and_cost(X, y, q, J_iter, num_iter, alpha)
print(f"theta0={q[0]}    theta1={q[1]}")
# D
h = lambda x: q[0] + x * q[1]
print(f"the predictable calories consumption for 250 kg mammal is: {(np.exp(h(np.log(250))) / 4.18) * 1000}")
# E
w = lambda x: (x - q[0]) / q[1]
print(f'The weight of mammal that consume 3.5 kjoul per day is {np.exp(w(np.log(3.5)))}')
