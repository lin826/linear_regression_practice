
import numpy
from numpy import ones
# Do Bayssian approach
def get_theta_Bayssian(phi,train_t,beta,alpha,m0):
    m0 = ones(shape=(len(phi),1))*m0
    phi_dot = phi.dot(phi.T)
    S0_ = 1/alpha*numpy.identity(len(phi_dot))
    Sn = numpy.linalg.inv(S0_+beta*phi_dot)
    mn = Sn.dot(S0_.dot(m0) + beta*phi.dot(train_t))
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
            if lamb != 0:
                regularization = (weights**2).sum()/len(weights) * lamb
            else:
                regularization = 0
            j_ = min(n, j + batch_size)
            phi_batch = phi[:,j:j_]
            t_batch = train_t[j:j_]
            # print('phi',phi.shape)
            lost = t_batch - numpy.dot(weights.T,phi_batch).T + regularization
            # print('lost',lost.shape)
            weights = weights + eta * phi_batch.dot(lost)
            # print('weights',weights.shape)
            # print(i,': ',weights)
        err = MSE(train_t - numpy.dot(weights.T,phi).T)
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

def evaluate_algorithm(test_x, test_t):
    phi = get_phi(test_x)
    return MSE(test_t - numpy.dot(weights.T,phi).T)

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
