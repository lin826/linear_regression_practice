import numpy as np
from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

x_min = 0
x_max = 100
y_min = 0
y_max = 100

def phi(x):
    return x

def model(x):
    return 1.0 * np.exp( - (np.square((x[:,0] - x_max/2))+np.square((x[:,1] - y_max/2))) / (2.0 * 20**2) )
    # return np.sqrt(np.square(x[:,0]-x_max/2) + np.square(x[:,1] - y_max/2))


X = np.arange(x_min, x_max, 1)
Y = np.arange(y_min, y_max, 1)
X, Y = np.meshgrid(X, Y)

X_flat, Y_flat = X.T.flatten(), Y.T.flatten()
Z = model(np.matrix(phi(np.array([X_flat, Y_flat], dtype=np.float32).T)))
print(Z.shape)
Z = np.reshape(Z, [len(X), len(Y)])
Z = np.rot90(Z)
plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), aspect = 'auto')
plt.colorbar()
plt.show()
