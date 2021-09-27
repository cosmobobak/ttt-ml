import numpy as np

x_val = np.array([
    np.array([np.zeros(9), np.zeros(9)]),
    np.array([np.array([0, 0, 1, 0, 1, 1, 0, 0, 0]),
              np.array([0, 1, 0, 0, 0, 0, 1, 0, 1])]),
    np.array([np.array([1, 1, 0, 1, 0, 0, 0, 0, 0]),
              np.array([0, 0, 1, 0, 0, 1, 0, 0, 0])]),
])

y_val = np.array([
    0,
    1,
    -1
])
