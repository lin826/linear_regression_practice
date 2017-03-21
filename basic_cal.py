
import numpy
from numpy import ones,zeros

sum_MSE = 0

# Do Bayesian approach
def get_theta_Bayesian(phi,train_t,beta,alpha,m0):
    m0 = ones(shape=(len(phi.T),1))*m0
    phi_dot = phi.T.dot(phi)
    S0_ = 1.0/alpha*numpy.identity(len(phi_dot))
    Sn = numpy.linalg.inv(S0_+beta*phi_dot)
    mn = Sn.dot(S0_.dot(m0) + beta*phi.T.dot(train_t))
    return mn

# Do Stochastic (sequential) gradient descent
# Return result_weights
def gradient_descent(train_x, train_t, eta, num_iters,lamb,batch_size):
    result_err = 1e+300
    n = len(train_x)
    phi = get_phi(train_x)
    weights = ones(shape=(len(Gausian_mean), 1))
    for i in range(num_iters):
        for j in range(0, n, batch_size):
            j_ = min(n, j + batch_size)
            regularization = zeros(shape=(j_-j, 1))
            if lamb != 0:
                r = numpy.square(weights).sum()/len(weights) * lamb
                regularization = ones(shape=(j_-j, 1))*r
            phi_batch = phi[j:j_]
            t_batch = train_t[j:j_]
            lost = t_batch - numpy.dot(phi_batch,weights) + regularization
            weights = weights + eta * phi_batch.T.dot(lost)
        err = MSE(train_t - numpy.dot(weights.T,phi.T).T)
        if(err < result_err):
            result_err = err
            result_weights = weights
    MSE(train_t - numpy.dot(result_weights.T,phi.T).T)
    return result_weights

def get_phi(X):
    phi = []
    for x in X:
        phi.append(Gausian_function(x))
    return numpy.array(phi)

def Gausian_function(x):
    result = []
    for m in Gausian_mean:
        a = 1.0 * numpy.exp( - (numpy.square(x[0] - m[0])+numpy.square(x[1] - m[1])) / (2.0 * Gausian_sigma**2) )
        result.append(a)
    return numpy.array(result)

def evaluate_algorithm(test_x, test_t):
    phi = get_phi(test_x)
    return MSE(test_t - numpy.dot(weights.T,phi).T)

# Get MSE error value
def MSE(lost):
    global sum_MSE
    mse = numpy.square(lost).sum()/len(lost)
    print('training gradient MSE: ',mse)
    sum_MSE += mse
    return mse

def get_sum_MSE():
    global sum_MSE
    return sum_MSE

def get_phi_MSE(X,y,weights):
    return (numpy.square(y - weights.T * get_phi(X))).sum()/len(y)

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
