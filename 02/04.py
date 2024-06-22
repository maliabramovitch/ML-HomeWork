import matplotlib.pyplot as plt
from sklearn import datasets

""" A """
iris = datasets.load_iris()
X = iris.data[:, 0:3]  # we only take the first two features.
x1 = X[:, 0].reshape((-1, 1))
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
