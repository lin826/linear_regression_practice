import numpy
import matplotlib.pyplot as plt
from numpy import loadtxt, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def show_3d_gragh(X,y,MAP_SIZE):
    a = numpy.arange(0, MAP_SIZE, 1)
    b = numpy.arange(0, MAP_SIZE, 1)
    a, b = numpy.meshgrid(a, b)
    result = [[0 for j in range(MAP_SIZE)] for i in range(MAP_SIZE)]
    # Insert the known data
    for e in range(len(y)):
        x1,x2 = int(X[e][0]),int(X[e][1])
        result[x1][x2] = y[e]
    fig = plt.figure(num=None, figsize=(10, 3), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(a, b, result, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def show_2d_gragh(X,y,MAP_SIZE):
    print('Start draw 2d graph')
    result = numpy.array([[0 for j in range(MAP_SIZE)] for i in range(MAP_SIZE)])
    # Insert the known data
    for e in range(len(y)):
        x1,x2 = int(X[e][0]),int(X[e][1])
        result[x1][x2] = y[e]
    # Show the Image
    plt.imshow(result, extent=(0, MAP_SIZE, 0, MAP_SIZE), aspect = 'auto')
    plt.colorbar()
    plt.show()
