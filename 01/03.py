import numpy as np
import matplotlib.pyplot as plt

def show_data():
    data = np.loadtxt('faithful.txt')
    n_row = data.shape[0]
    x = data[:, 0].reshape(n_row, 1)
    y = data[:, 1].reshape(n_row, 1)
    plt.plot(x, y, 'xr')
    plt.title('Faithful Geiger')
    plt.xlabel('Duration of minutes of the eruption')
    plt.ylabel('Time to next eruption (minutes)')
    plt.show()

show_data()
