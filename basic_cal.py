
import numpy
from numpy import ones
# Do Stochastic (sequential) gradient descent
# Return result_weights
def gradient_descent(train_x, train_t, eta, num_iters):
    result_err = 1e+300
    phi = get_phi(train_x)
    weights = ones(shape=(len(Gausian_mean), 1))
    for i in range(num_iters):
        lost = train_t - numpy.dot(weights.T,phi).T
        weights = weights + eta * phi.dot(lost)
        print(i,': ',weights)
        err = MSE(lost)
        if(err < result_err):
        	result_err = err
        	result_weights = weights

    return result_weights

def get_phi(X):
    phi = []
    for x in X:
        phi.append(Gausian_function(x))
    return numpy.array(phi).T

def Gausian_function(x):
    result = []
    for m in Gausian_mean:
        a = 1 * numpy.exp( - ((x[0] - m[0])**2+(x[1] - m[1])**2) / (2 * Gausian_sigma**2) )
        result.append(a)
    return numpy.array(result)

# Get MSE error value
def MSE(lost):
    mse = (lost**2).sum()/len(lost)
    print('mean square error: ',mse)
    return mse

# Two ways to get mean in Gausian: grid_mean, k_mean
def set_Gausian_mean(BASIS_SIZE,MAP_SIZE,mean_get):
    global Gausian_mean
    Gausian_mean =  mean_get(BASIS_SIZE,MAP_SIZE)

def set_Gausian_sigma(SIGMA):
    global Gausian_sigma
    Gausian_sigma = SIGMA

def grid_mean(BASIS_SIZE,MAP_SIZE):
    mean = []
    l = int(BASIS_SIZE**0.5)
    for i in range(l):
        for j in range(l):
            x = (i+1)*MAP_SIZE/(l+1)
            y = (j+1)*MAP_SIZE/(l+1)
            mean.append([x,y])
    return numpy.array(mean)

def k_mean(settings):
    ## TODO: Implement k_mean method to get mean value
    return 0
