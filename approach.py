import numpy
from draw_graph import *
from data_io import *
from basic_cal import *

def model_init(s):
    global Ground_x,Ground_t,Train_x,Train_t,Test_x,Test_t,settings
    Train_x, Train_t = get_train_data(s['x_train'],s['t_train'])
    n = len(Train_x)
    indices = numpy.asarray(range(n), dtype=numpy.int32)
    n_train = s['data_size']
    numpy.random.shuffle(indices)
    Ground_x = numpy.array(Train_x)
    Ground_t = numpy.array(Train_t)
    Ground_x = Ground_x[indices[:n_train]]
    Ground_t = Ground_t[indices[:n_train]]
    settings = s
    print('Finish initializing')

def model_setting(s,k):
    global Train_x,Train_t,Test_x,Test_t
    if s['k_folder'] == 0:
        Train_x = Ground_x
        Train_t = Ground_t
        Test_x = Ground_x
        Test_t = Ground_t
    else:
        if(k>s['k_folder']):
            return -1
        k_size = int(s['data_size'] / s['k_folder'])
        Train_x = Ground_x[k_size*(k-1) :k_size*(k)]
        Train_t = Ground_t[k_size*(k-1) :k_size*(k)]
        Test_x = Ground_x[k_size*(k):k_size*(k+1)]
        Test_t = Ground_t[k_size*(k):k_size*(k+1)]
        print(k,':',Train_t.shape,Test_t.shape)
    return 0

def set_Gausian_basis():
    set_Gausian_mean(settings['basis_size'],settings['map_size'],grid_mean)
    set_Gausian_sigma(settings['sigma'])

# Test on Bayessian now
def Test_approach():
    phi = get_phi(Train_x)
    weights = get_theta_Bayessian(phi,Train_t,settings['beta'],settings['alpha'],settings['m0'])
    MSE(Train_t - numpy.dot(weights.T,phi).T)
    return weights

def Test_approach():
    phi = get_phi(Train_x)
    weights = get_theta_Bayessian(phi,Train_t,settings['beta'],settings['alpha'],settings['m0'])
    MSE(Train_t - numpy.dot(weights.T,phi).T)
    return weights

def MAP_approach():
    print('Start MAP approach')
    weights = gradient_descent(Train_x,Train_t, settings['eta'], settings['iter'],settings['lamb'],settings['batch_size'])
    return weights

def ML_approach():
    print('Start ML approach')
    weights = gradient_descent(Train_x,Train_t, settings['eta'], settings['iter'],0,settings['batch_size'])
    return weights

def draw(weights,method):
    graph_x,graph_t = get_graph_data(weights)
    method(graph_x,graph_t,settings['map_size'])

def get_graph_data(w):
    s = settings['map_size']
    scale = 2
    grid_size = int(s/scale)
    graph_x = [[i*scale,j*scale] for j in range(grid_size) for i in range(grid_size)]
    print('Start get graph_t')
    phi = get_phi(graph_x)
    graph_t = numpy.dot(w.T,phi).T
    print(graph_t)
    return numpy.array(graph_x),numpy.array(graph_t)
