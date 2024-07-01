# -*- coding: utf-8 -*-
"""
Created on Sat May 21 23:59:03 2022

@author: User
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def dist_neigh(X, y, xj, k):
    diff = (X - xj)
    d = np.sum(diff * diff, axis=1) ** (1 / 2)
    k_nearest = np.argsort(d)[:k]
    return d, y[list(k_nearest)]


def knn_classifier_single(X, y, xj, k):
    d, k_nearest_index = dist_neigh(X, y, xj, k)
    values, counts = np.unique(k_nearest_index, return_counts=True)
    max_counts_index = np.argmax(counts)
    max_counts = counts[max_counts_index]
    if counts[counts == max_counts].shape[0] > 1:
        return np.argmin([d[y == 0].sum(), d[y == 1].sum(), d[y == 2].sum()])
    else:
        return values[max_counts_index]


def knn_classifier(X_train, y_train, X_test, y_test, attributes_cols, k):
    correct_predictions = 0
    y_predict = np.zeros(y_test.shape)
    for i in range(len(X_test)):
        y_predict[i, 0] = knn_classifier_single(X_train[:, attributes_cols], y_train, X_test[i, attributes_cols], k)
        if y_test[i, 0] == y_predict[i, 0]:
            correct_predictions = correct_predictions + 1
    return correct_predictions, int((correct_predictions / X_test.shape[0]) * 100), y_predict


""" A """
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
penguins = pd.read_csv(
    "https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")

penguins = penguins.dropna()
penguins.species_short.value_counts()
penguins.head()
penguins['species_short'].value_counts()

sns.pairplot(penguins, hue='species_short')
penguin_data = penguins[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values
scaled_penguin_data = StandardScaler().fit_transform(penguin_data)
y_labels = penguins.species_short.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})
y = y_labels.to_numpy()
y1 = y.reshape(y.shape[0], 1)

""" B """
X_train = np.concatenate((scaled_penguin_data[(y == 0), :][:100, :], scaled_penguin_data[(y == 1), :][:50, :],
                          scaled_penguin_data[(y == 2), :][:80, :]), axis=0)
y_train = np.concatenate((y[y == 0][:100], y[(y == 1)][:50], y[(y == 2)][:80]), axis=0)
X_test = np.concatenate((scaled_penguin_data[(y == 0), :][100:, :], scaled_penguin_data[(y == 1), :][50:, :],
                         scaled_penguin_data[(y == 2), :][80:, :]), axis=0)
y_test = np.concatenate((y[y == 0][100:], y[(y == 1)][50:], y[(y == 2)][80:]), axis=0).reshape((-1, 1))

print("2 attributes")
for k in [1, 3, 5]:
    correct_predictions, percentage, y_predict = knn_classifier(X_train, y_train, X_test, y_test,
                                                                (0, 2), k)
    print(f"k = {k}", f"The number of samples that classified correct: {correct_predictions}",
          f"the percentage of right identification is {percentage}%", sep='\n', end='\n\n')
print("4 attributes")
for k in [1, 3, 5]:
    correct_predictions, percentage, y_predict = knn_classifier(X_train, y_train, X_test, y_test, (range(0, 4)), k)
    print(f"k = {k}", f"The number of samples that classified correct: {correct_predictions}",
          f"the percentage of right identification is {percentage}%", sep='\n', end='\n\n')
