import numpy as np


def square_euclidean_distance(data):
    def distance(v1, v2):
        return np.sqrt(((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2))

    distances = []
    for v1 in data:
        row = []
        for v2 in data:
            row.append(distance(v1, v2))
        distances.append(row)
    return distances


def print_mat(mat):
    print("   ", end="")
    for i in range(len(mat)):
        print(f"{i}     ", end='')
    print()
    for i in range(len(mat)):
        print(f"{i}  ", end='')
        for j in range(len(mat)):
            print(f"{mat[i][j]:.3f} ", end='')
        print()


Dx = [[1, 1], [2, 1], [5, 4], [6, 5], [6.5, 6]]
print_mat(square_euclidean_distance(Dx))
