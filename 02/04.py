import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


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


def plot_log_reg_line(X, y, theta, title='data', x1_title='x1', x2_title='x2'):
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
    plt.xlabel(x1_title), plt.ylabel(x2_title), plt.title(title),
    plt.grid(axis='both'), plt.show()


def test_prediction(classifiers, X_samples, rigth_tags):
    test_predictions = []
    correct_predictions = 0
    for i in range(len(X_samples)):
        chances = np.array([sigmoid(np.dot(X_samples[i], theta)) for theta in classifiers])
        prediction = chances.argmax()
        test_predictions.append(prediction)
        if prediction == rigth_tags[i]:
            correct_predictions += 1
    test_predictions = np.array(test_predictions)
    rigth_tags_row = rigth_tags.reshape((1, rigth_tags.size))[0]
    print(f"predictios = {test_predictions}")
    print(f"tests tags = {rigth_tags_row}")
    print()
    print(f"the percentage of right identification is {int(correct_predictions / rigth_tags.size * 100)}%", end='\n\n')


""" A """
iris = datasets.load_iris()
X = iris.data[:, 0:4]  # we only take the first two features.
x1 = X[:, 1].reshape((-1, 1))
x2 = X[:, 2].reshape((-1, 1))
plt.figure(2, figsize=(8, 6))
plt.clf()
setosa = plt.scatter(x1[0:35, 0], x2[0:35, 0], color='red')
versicolor = plt.scatter(x1[51:86, 0], x2[51:86, 0], color='green')
virginica = plt.scatter(x1[100:135, 0], x2[100:135, 0], color='blue')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.legend((setosa, versicolor, virginica),
           ('setosa', 'versicolor', 'virginica'),
           scatterpoints=1,
           loc='lower right',
           ncol=1,
           fontsize=16)
plt.show()

""" B """
X_petal_length_sepal_width = X[:, 1:3]
test_sampels_indexes = np.concatenate([np.arange(0, 35), np.arange(50, 85), np.arange(100, 135)])
X_samples_orig = X_petal_length_sepal_width[test_sampels_indexes, :]
m = X_samples_orig.shape[0]
onesvec = np.ones((m, 1))
X_samples = np.concatenate((onesvec, X_samples_orig), axis=1)
n = X_samples.shape[1]

# Setosa
y_setosa = np.array([1 if i < 35 else 0 for i in test_sampels_indexes]).reshape((-1, 1))
theta_setosa = np.zeros((n, 1))
theta_setosa, J_iter_setosa = theta, J_iter = gradDescent_log(X_samples, y_setosa, theta_setosa, 0.1, 10000)

# Versicolor
y_versicolor = np.array([1 if 50 <= i < 86 else 0 for i in test_sampels_indexes]).reshape((-1, 1))
theta_versicolor = np.zeros((n, 1))
theta_versicolor, J_iter_versicolor = gradDescent_log(X_samples, y_versicolor, theta_versicolor, 0.1, 10000)

# Virginica
y_virginica = np.array([1 if 100 <= i < 135 else 0 for i in test_sampels_indexes]).reshape((-1, 1))
theta_virginica = np.zeros((n, 1))
theta_virginica, J_iter_virginica = gradDescent_log(X_samples, y_virginica, theta_virginica, 0.4, 15000)

''' C '''
plot_log_reg_line(X_samples, y_setosa, theta_setosa, "Setosa vs all", 'petal length', 'sepal width')
plot_log_reg_line(X_samples, y_versicolor, theta_versicolor, "Versicolor vs all", 'petal length', 'sepal width')
plot_log_reg_line(X_samples, y_virginica, theta_virginica, "Virginica vs all", 'petal length', 'sepal width')

''' D '''
X_petal_length_sepal_width = X[:, 1:3]
test_sampels_indexes = np.concatenate([np.arange(35, 50), np.arange(85, 100), np.arange(135, 150)])
X_samples_orig = X_petal_length_sepal_width[test_sampels_indexes, :]
m = X_samples_orig.shape[0]
onesvec = np.ones((m, 1))
X_samples = np.concatenate((onesvec, X_samples_orig), axis=1)
y_setosa_b = np.array([0 for i in test_sampels_indexes if 35 <= i < 50]).reshape((-1, 1))
y_versicolor_b = np.array([1 for i in test_sampels_indexes if 85 <= i < 100]).reshape((-1, 1))
y_virginica_b = np.array([2 for i in test_sampels_indexes if 135 <= i < 150]).reshape((-1, 1))
rigth_tags = np.concatenate((y_setosa_b, y_versicolor_b, y_virginica_b))
classifiers = [theta_setosa, theta_versicolor, theta_virginica]

test_prediction(classifiers, X_samples, rigth_tags)

""" E """
X_petal_length_sepal_width = X
test_sampels_indexes = np.concatenate([np.arange(0, 35), np.arange(50, 85), np.arange(100, 135)])
X_samples_orig = X_petal_length_sepal_width[test_sampels_indexes, :]
m = X_samples_orig.shape[0]
onesvec = np.ones((m, 1))
X_samples = np.concatenate((onesvec, X_samples_orig), axis=1)
n = X_samples.shape[1]

# Setosa
y_setosa = np.array([1 if i < 35 else 0 for i in test_sampels_indexes]).reshape((-1, 1))
theta_setosa = np.zeros((n, 1))
theta_setosa, J_iter_setosa = theta, J_iter = gradDescent_log(X_samples, y_setosa, theta_setosa, 0.1, 10000)

# Versicolor
y_versicolor = np.array([1 if 50 <= i < 86 else 0 for i in test_sampels_indexes]).reshape((-1, 1))
theta_versicolor = np.zeros((n, 1))
theta_versicolor, J_iter_versicolor = gradDescent_log(X_samples, y_versicolor, theta_versicolor, 0.1, 10000)

# Virginica
y_virginica = np.array([1 if 100 <= i < 135 else 0 for i in test_sampels_indexes]).reshape((-1, 1))
theta_virginica = np.zeros((n, 1))
theta_virginica, J_iter_virginica = gradDescent_log(X_samples, y_virginica, theta_virginica, 0.4, 15000)

X_petal_length_sepal_width = X
test_sampels_indexes = np.concatenate([np.arange(35, 50), np.arange(85, 100), np.arange(135, 150)])
X_samples_orig = X_petal_length_sepal_width[test_sampels_indexes, :]
m = X_samples_orig.shape[0]
onesvec = np.ones((m, 1))
X_samples = np.concatenate((onesvec, X_samples_orig), axis=1)
y = iris.target
y_setosa_b = np.array([0 for i in test_sampels_indexes if 35 <= i < 50]).reshape((-1, 1))
y_versicolor_b = np.array([1 for i in test_sampels_indexes if 85 <= i < 100]).reshape((-1, 1))
y_virginica_b = np.array([2 for i in test_sampels_indexes if 135 <= i < 150]).reshape((-1, 1))
rigth_tags = np.concatenate((y_setosa_b, y_versicolor_b, y_virginica_b))
classifiers = [theta_setosa, theta_versicolor, theta_virginica]
print()
test_prediction(classifiers, X_samples, rigth_tags)
