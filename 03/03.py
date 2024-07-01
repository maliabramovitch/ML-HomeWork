import numpy as np


def d(x1, x2, p):
    return ()


def init_centroid(k, dataset):
    centroids_indexes = np.random.choice(dataset.shape[0], size=k, replace=False)
    return dataset[centroids_indexes, :], centroids_indexes


def assign_samples(k, dataset):
    centroids, centroids_indexes = init_centroid(dataset, k)
    assignment = np.zeros((dataset.shape[0], 1))



def centroid_calc(k, dataset):
    pass


def k_means(k, dataset, threshold):
    pass
