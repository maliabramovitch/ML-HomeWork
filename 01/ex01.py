import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set()  # data visualization library

# preparing the data
a1 = 1.8
a0 = -2
x = 10 * np.random.rand(100)
y = a0 + a1 * x + np.random.randn(100)
plt.scatter(x, y)

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 10000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
print("Model slope a1 = ", model.coef_[0])
print("Model intercept a0 = ", model.intercept_)
plt.show()

# 500 samples
a1 = 2.7
a0 = 5
x = 35 * np.random.rand(500)
y = a0 + a1 * x + np.random.normal(0, 25, 500)
plt.scatter(x, y)

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 35, 10000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
print("Model slope a1 = ", model.coef_[0])
print("Model intercept a0 = ", model.intercept_)
plt.show()
