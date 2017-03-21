import numpy
import matplotlib.pyplot as plt
from numpy import loadtxt, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def show_3d_gragh(X,y,MAP_SIZE,scale):
    a = numpy.arange(0, MAP_SIZE, scale)
    b = numpy.arange(0, MAP_SIZE, scale)
    a, b = numpy.meshgrid(a, b)
    steps = int(MAP_SIZE/scale)+1
    result = [[0 for j in range(steps)] for i in range(steps)]
    # Insert the known data
    for e in range(len(y)):
        x1,x2 = int(X[e][0]/scale),int(X[e][1]/scale)
        result[x1][x2] = y[e][0]
    fig = plt.figure(num=None, figsize=(10, 3), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(a, b, result, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def show_2d_gragh(X,y,MAP_SIZE,scale):
    print('Start draw 2d graph')
    steps = int(MAP_SIZE/scale)
    result = numpy.array([[0 for j in range(steps)] for i in range(steps)])
    # Insert the known data
    for e in range(len(y)):
        x1,x2 = int(X[e][0]/scale),int(X[e][1]/scale)
        result[x1][x2] = y[e][0]
    # Show the Image
    plt.imshow(result, extent=(0, MAP_SIZE, 0, MAP_SIZE), aspect = 'auto')
    plt.colorbar()
    plt.show()
