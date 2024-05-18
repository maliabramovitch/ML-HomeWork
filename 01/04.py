import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_reg_line(X, y, theta):
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
    y_min = y[ind_min] * 0.9
    y_max = y[ind_max] * 1.1
    Xlh = X[(ind_min, ind_max), :]
    yprd_lh = np.dot(Xlh, theta)
    plt.plot(X[:, ind], y, 'go', Xlh, yprd_lh, 'm-')
    space_x = ((x_max - x_min) / 100) * 20
    space_y = ((y_max - y_min) / 100) * 20
    plt.axis((x_min - space_x, x_max + space_x, min(y_min, y_max) - space_y, max(y_min, y_max) + space_y))
    plt.xlabel('x'), plt.ylabel('y'),
    plt.title(f'Intercept: {theta[0, 0]}  -  Slope: {theta[1, 0]}', color='red')
    plt.suptitle('My model')
    plt.grid()
    plt.show()


def show_data():
    xs = np.load('TA_Xhouses.npy')
    print(xs)
    print()
    ys = np.load('TA_yprice.npy')
    print(ys)
    plt.plot(xs, ys, 'xr')
    plt.title('Cost of house in 100K ILS')
    plt.xlabel("House's front Length in meters")
    plt.ylabel("House's cost in 100K ILS")
    plt.show()
    return xs, ys


def cost_computation(X, y, q):
    m = y.shape[0]
    z = np.dot(X, q) - y
    J = 1 / (2 * m) * (np.dot(z.T, z)[0][0])
    return J


def gd_batch(X, y, q, alpha, num_iter):
    J_iter = np.zeros((num_iter, 1))
    for j in range(num_iter):
        q -= alpha * (X.T @ (np.dot(X, q) - y))
        J_iter[j] = cost_computation(X, y, q)
    return q, J_iter


x, y = show_data()
if x.shape[1] == 2:
    ind = 1
else:
    ind = 0

x_min = x[:, ind].min() - 1
x_max = x[:, ind].max() + 1
model = LinearRegression(fit_intercept=True)
theta = np.random.normal(x.size)
model.fit(x, y)
xfit = np.linspace(x_min, x_max, 10000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.title("LinearRegression by sklearn.linear_model")
plt.show()
theta = np.zeros((2, 1))
m = x.shape[0]
X = np.concatenate((np.ones((m, 1)), x), axis=1)
q, J_iter = gd_batch(X, y, theta, 0.0001, 5000)
plot_reg_line(X, y, theta)
a = model.coef_[0]
b = model.intercept_
y = lambda x: x*a + b
for m in [15, 27]:
    print(f'The predicted price for {m} meters = {y(m)[0]}')