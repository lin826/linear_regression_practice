# Reference:
# http://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/


# Standalone simple linear regression example
from math import sqrt

# Calculate mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
	print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse

# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions

# Test simple linear regression
# dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# rmse = evaluate_algorithm(dataset, simple_linear_regression)
# print('RMSE: %.3f' % (rmse))

import numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import sqrt

sigma = 100
MAP_SIZE = 1081

a = np.arange(0, MAP_SIZE, 1)
b = np.arange(0, MAP_SIZE, 1)
a, b = np.meshgrid(a, b)
result = [[-1 for j in range(MAP_SIZE)] for i in range(MAP_SIZE)]

mean_sigma = [[0,300,100],[500,300,100],[800,300,100]]
for i in range(MAP_SIZE):
	for j in range(MAP_SIZE):
		v = 0
		for k in range(len(mean_sigma)):
			v += np.exp(- ((i - mean_sigma[k][0])**2+(j - mean_sigma[k][1])**2) / (2 * mean_sigma[k][2]**2))
		result[i][j] = v
fig = plt.figure(num=None, figsize=(10, 3), dpi=200, facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(a, b, result, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
