import numpy
import csv
import matplotlib.pyplot as plt
from numpy import loadtxt, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import sqrt
from simple_linear_regression import *


DATA_SIZE = 40000
MAP_SIZE = 1082
DIM = 2
num_basis = 225

def set_X_Y(DATA_SIZE):
    with open('data/X_train.csv', 'r') as locCsvfile:
        loc_reader = csv.reader(locCsvfile)
        train_loc = ones(shape=(DATA_SIZE, 2))
        i = 0
        for row in loc_reader:
            train_loc[i] = [int(iterator) for iterator in row]
            i += 1
            if(i>=DATA_SIZE): break
    with open('data/T_train.csv', 'r') as valCsvfile:
        val_reader = csv.reader(valCsvfile)
        train_val = ones(shape=(DATA_SIZE, 1))
        i = 0
        for row in val_reader:
            train_val[i] = int(row[0])
            i += 1
            if(i>=DATA_SIZE): break
    return train_loc, train_val[:,0]

# Calculate mean squared error
def mse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual)*2)
	return mean_error

def evaluate_algorithm(data, value, theta):
	predicted = get_predict_value(data,theta)
	err = mse_metric(value, predicted)
	print('theta: ',theta)
	print('mean square error: ',err)
	return predicted,err

def show_gragh(X,y,theta):
    a = numpy.arange(0, MAP_SIZE, 1)
    b = numpy.arange(0, MAP_SIZE, 1)
    a, b = numpy.meshgrid(a, b)
    it = ones(shape=(MAP_SIZE, DIM))
    result = [[-1 for j in range(MAP_SIZE)] for i in range(MAP_SIZE)]
    for e in range(len(y)):
        x1,x2 = int(X[e][0]),int(X[e][1])
        result[x1][x2] = y[e]
    for i in range(MAP_SIZE):
        it[:, 0] = a[i]
        it[:, 1] = b[i]
        r = get_predict_value(it,theta).flatten()
        for j in range(MAP_SIZE):
            if(result[i][j]<0):
                result[i][j] = r[j]
        	# result[i][j] = r[j]

#Load the dataset
X,y= set_X_Y(DATA_SIZE)
#number of training samples
m = y.size
#Add a column of ones to X (interception data)
it = ones(shape=(m, DIM))
for i in range(DIM):
    it[:, i] = X[:,i]
#Initialize theta parameters
theta = ones(shape=(num_basis, 1))
theta = numpy.array([[  4.24379901e-03], [ -3.81816533e-03], [  1.74801200e-03], [ -7.13105335e-03], [ -7.81772079e-03], [  7.86142360e-04], [  4.55935268e-02], [  1.43373516e-02], [ -6.01095581e-02], [ -2.36382810e-03], [  2.91763216e-02], [  1.16335017e-02], [ -6.65292104e-03], [ -4.63885660e-03], [  2.05650420e-02], [  1.85487014e-03], [  4.61472687e-03], [  2.79455343e-02], [  3.40380870e-02], [  6.44806511e-02], [  7.18182631e-02], [ -9.61337469e-02], [ -2.49041244e-01], [  2.36986451e-01], [ -9.80334383e-02], [ -1.64578611e-02], [ -2.05169323e-02], [  8.26060593e-03], [  1.84482961e-02], [  7.80490039e-02], [  1.00522243e-02], [ -1.52715603e-03], [ -7.56928939e-02], [ -1.65363411e-01], [ -2.27242977e-01], [ -1.95596959e-01], [ -8.69069597e-02], [  9.47099313e-01], [  4.66519764e-01], [  3.75274534e-02], [  4.96269634e-02], [  1.72175710e-03], [  1.06108273e-01], [  7.35194473e-02], [ -1.57226117e-01], [  5.14334233e-03], [ -5.93423159e-02], [  1.14518745e-01], [  4.57460620e-01], [  4.90800183e-01], [  2.60953901e-01], [  4.49347990e-01], [ -7.05210633e-02], [ -1.21634167e-01], [  2.23203618e-02], [  4.56570468e-02], [ -2.51770910e-02], [  8.76197056e-02], [ -1.02511238e+00], [  5.21192140e-01], [ -2.16453352e-02], [  7.04950368e-02], [  2.50732772e-01], [  2.13273198e-01], [  1.37937736e-02], [ -2.65974026e-02], [ -7.64580773e-03], [  9.01943789e-03], [  4.81119610e-02], [ -6.98273672e-02], [  6.62036956e-02], [ -9.76836586e-02], [  6.13645366e-01], [  1.28944946e+00], [  4.13281811e+00], [  2.88868391e-03], [ -4.14178220e-02], [ -8.08927232e-02], [ -1.10429293e-01], [ -8.14477533e-02], [ -2.16801015e-01], [ -1.09131949e-01], [ -9.14344169e-02], [ -8.27847384e-02], [  6.08450632e-01], [  2.89640028e-01], [  3.88621141e-01], [  2.88214901e-01], [ -6.98313224e-01], [  4.53827448e-01], [  1.14312953e-03], [  1.32765324e-02], [  5.24717429e-02], [ -1.53866712e-01], [  6.83084654e-01], [  1.16918312e+00], [  4.79728743e-01], [  6.58239027e-01], [  1.65192212e+00], [ -9.80317907e-01], [  1.78389582e+00], [ -3.71909144e-03], [  5.25376229e-01], [  3.46936280e-01], [  5.34126576e-01], [ -1.13610466e-02], [  1.78509792e-02], [  1.32394748e-02], [ -5.40407361e-02], [  1.49021489e-01], [  6.48052270e-01], [  3.00108091e-01], [ -9.29540711e-02], [  2.34175664e+00], [  2.47468696e+00], [  3.39717450e-03], [  1.65036616e-01], [  1.65736531e-01], [  1.22874074e+00], [  6.77801024e-01], [  4.99535030e-03], [  4.78067364e-02], [ -9.33230654e-02], [ -3.78671914e-02], [  8.21140042e-01], [  1.48488925e-01], [  4.83346523e-02], [ -2.91409298e-01], [  1.12880359e-01], [  2.62806356e+00], [  1.42669302e-01], [  2.18706336e+00], [  2.85360053e+00], [  8.70584933e-01], [  1.34203641e+00], [  5.11365854e-02], [ -6.31492982e-02], [ -2.31305379e-01], [  7.08401954e-01], [  1.37070932e+00], [  7.24557865e-01], [ -2.11129055e-01], [  2.60037637e-01], [  7.43029845e-01], [ -1.24805146e-01], [ -3.24047075e-01], [ -7.16134245e-01], [  4.79179295e-01], [  4.42229903e-01], [  2.10642609e+00], [ -1.42321660e-01], [ -1.10103738e-01], [  7.03070805e-01], [  1.48844483e-01], [  4.00865470e-01], [ -5.47649792e-02], [  1.23771379e-01], [ -1.43728511e-01], [ -7.37362611e-02], [ -1.83360817e-01], [  1.53262966e+00], [  7.43903271e-01], [  5.53947542e-01], [  2.02202861e-01], [  2.46695384e+00], [  1.05727514e-01], [  6.75979337e-01], [ -2.06992915e-01], [ -2.13969081e-01], [ -1.34738769e-01], [ -4.73020652e-01], [  1.29453077e+00], [  9.79062765e-01], [ -7.67461217e-02], [  1.35658552e-01], [  9.41222549e-01], [  1.94057429e+00], [  7.69652005e-01], [ -3.71815734e-01], [  1.47025767e-01], [ -4.36229194e-02], [ -1.55648675e-01], [ -3.11684358e-02], [  7.58689436e-02], [  1.23472151e-01], [ -1.12656251e-01], [ -2.09859885e-01], [  1.25462801e+00], [  1.22449167e+00], [  1.55048409e+00], [  5.74059359e-01], [  1.18688290e-01], [  1.69873291e+00], [  5.73092732e-01], [  7.50526719e-01], [  1.40934631e-02], [  1.67273574e-02], [  2.15210412e-02], [ -2.35272375e-02], [ -3.74219501e-04], [  4.44826364e-02], [ -8.82847359e-02], [ -6.39511498e-01], [  3.82304795e-01], [  6.61536445e-01], [  7.59705371e-01], [  1.33832511e+00], [  6.89731124e-01], [  9.63974956e-01], [ -4.09510957e-01], [  6.38004706e-03], [ -1.76395838e-03], [  3.80507280e-03], [  2.76280617e-04], [ -8.11428341e-03], [  6.39342643e-03], [  9.65447834e-02], [  8.26152986e-02], [ -1.42495592e-01], [ -1.86013893e-01], [ -3.48340345e-01], [ -4.79806780e-01], [ -4.11089470e-01], [ -1.40153511e-01], [  3.38460477e-02]])


set_Gausian(it, y, theta,MAP_SIZE)
#compute and display initial cost
# evaluate_algorithm(it, y, theta)

#Train and get weights/theta
theta = get_theta_Bayssian(it,y,theta)

#compute and display final cost
evaluate_algorithm(it, y, theta)

#Show the 3D graph
show_gragh(X,y,theta)
