import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def penguin():
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
    return scaled_penguin_data, y


def distance(x1, x2):
    diff = x1 - x2
    return np.sqrt(np.sum((diff * diff), axis=1))


def init_centroid(k, dataset):
    centroids_indexes = np.random.choice(dataset.shape[0], size=k, replace=False)
    return dataset[centroids_indexes, :]


def assign_samples(dataset: np.array, centroids):
    m = dataset.shape[0]
    assignment = np.empty(m, dtype=int)
    for i in range(m):
        dis = distance(dataset[i, :], centroids)
        assignment[i] = np.argmin(dis)
    return assignment


def centroid_calc(k, dataset, assignment):
    centroids = np.zeros((k, dataset.shape[1]))
    for i in range(k):
        if np.any(assignment == i):
            centroids[i, :] = np.mean(dataset[assignment == i], axis=0)
        else:
            print(f"Cluster {i} has no points assigned!")
    return centroids


def calc_error(k, dataset, centroids, assignment):
    err = 0
    n = dataset.shape[0]
    for i in range(k):
        diff = dataset[assignment == i] - centroids[i]
        err += np.sum((diff * diff), axis=0)
    err /= n
    return np.sum(err, axis=0)


def num_clusters_prod(k, assignment, y):
    counts = np.zeros(k)
    for i in range(assignment.size):
        if assignment[i] == y[i]:
            counts[assignment[i]] += 1
    return counts


def k_means(k, dataset, threshold=0.0001, max_iter=1000):
    centroids = init_centroid(k, dataset)
    assignment = None
    last_err = 0
    for i in range(max_iter):
        assignment = assign_samples(dataset, centroids)
        centroids = centroid_calc(k, dataset, assignment)
        err = calc_error(k, dataset, centroids, assignment)
        if np.abs(err - last_err) < threshold:
            return centroids, assignment
        last_err = err
    return centroids, assignment


min_assignment = None
min_err = None
min_centroides = None
dataset, y = penguin()
for i in range(10):
    centroids, assignment = k_means(3, dataset[:, (0, 2)])
    err = calc_error(3, dataset[:, (0, 2)], centroids, assignment)
    if min_assignment is None or err < min_err:
        min_assignment = assignment
        min_err = err
        min_centroids = centroids
plt.figure(1)
plt.subplot(111)
plt.plot(dataset[min_assignment == 0, 0], dataset[min_assignment == 0, 2], 'go', label='Cluster 0')
plt.plot(dataset[min_assignment == 1, 0], dataset[min_assignment == 1, 2], 'ro', label='Cluster 1')
plt.plot(dataset[min_assignment == 2, 0], dataset[min_assignment == 2, 2], 'bo', label='Cluster 2')
plt.plot(min_centroids[0, 0], centroids[0, 1], 'gx', label='Centroid 0')
plt.plot(min_centroids[1, 0], centroids[1, 1], 'rx', label='Centroid 1')
plt.plot(min_centroids[2, 0], centroids[2, 1], 'bx', label='Centroid 2')
plt.show()
print("2 attributes")
d = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
for i in range(3):
    unique_elements, counts = np.unique(assignment[y == i], return_counts=True)
    print(f"{d[i]} class correct prediction: {np.max(counts)}/{assignment[y == i].size}")
print("\nall attributes")
for i in range(10):
    centroids, assignment = k_means(3, dataset)
    err = calc_error(3, dataset, centroids, assignment)
    if min_assignment is None or err < min_err:
        min_assignment = assignment
        min_err = err
        min_centroids = centroids
for i in range(3):
    unique_elements, counts = np.unique(assignment[y == i], return_counts=True)
    print(f"{d[i]} class correct prediction: {np.max(counts)}/{assignment[y == i].size}")
