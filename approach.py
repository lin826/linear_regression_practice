import numpy
from draw_graph import *
from data_io import *
from basic_cal import *

def model_init(s):
    global Ground_x,Ground_t,settings
    Ground_x, Ground_t = get_train_data(s['x_train'],s['t_train'])
    Ground_x = numpy.array(Ground_x)
    Ground_t = numpy.array(Ground_t)
    settings = s
    print('Finish initializing')

def model_setting(s,k):
    global Ground_x,Ground_t,Train_x,Train_t,Test_x,Test_t
    n = len(Ground_x)
    indices = numpy.asarray(range(n), dtype=numpy.int32)
    n_train = s['data_size']
    numpy.random.shuffle(indices)
    if s['k_folder'] == 0:
        Train_x = Ground_x[indices[:n_train]]
        Train_t = Ground_t[indices[:n_train]]
        Test_x = Ground_x[indices[n_train:]]
        Test_t = Ground_t[indices[n_train:]]
        print('test data size:',Test_t.shape)
    else:
        if(k>s['k_folder']):
            return -1
        k_size = int(s['data_size'] / s['k_folder'])
        Train_x = Ground_x[k_size*(k-1) :k_size*(k)]
        Train_t = Ground_t[k_size*(k-1) :k_size*(k)]
        if(k==s['k_folder']):
            Test_x = Ground_x[0:k_size*(1)]
            Test_t = Ground_t[0:k_size*(1)]
        else:
            Test_x = Ground_x[k_size*(k):k_size*(k+1)]
            Test_t = Ground_t[k_size*(k):k_size*(k+1)]
        print(k,':',Train_t.shape,Test_t.shape)
    print('Train_x',Train_x.shape)
    print('Train_t',Train_t.shape)
    return 0

def set_Gausian_basis():
    set_Gausian_mean(settings['basis_size'],settings['map_size'],grid_mean)
    set_Gausian_sigma(settings['sigma'])

# Test on Bayesian now
def Test_approach():
    phi = get_phi(Train_x)
    weights = get_theta_Bayesian(phi,Train_t,settings['beta'],settings['alpha'],settings['m0'])
    return weights

def Bayesian_approach():
    phi = get_phi(Train_x)
    weights = get_theta_Bayesian(phi,Train_t,settings['beta'],settings['alpha'],settings['m0'])
    return weights

def MAP_approach():
    print('Start MAP approach')
    weights = gradient_descent(Train_x,Train_t, settings['eta'], settings['iter'],settings['lamb'],settings['batch_size'])
    return weights

def ML_approach():
    print('Start ML approach')
    weights = gradient_descent(Train_x,Train_t, settings['eta'], settings['iter'],0,settings['batch_size'])
    return weights

def cal_MSE(weights):
    if(settings['k_folder']>0):
        print('Average MSE:',get_sum_MSE()/settings['k_folder'])
    else:
        print(get_phi_MSE(Test_x,Test_t,weights))

def draw(weights,method):
    graph_x,graph_t = get_graph_data(weights)
    method(graph_x,graph_t,settings['map_size'],settings['graph_scale'])

def get_graph_data(w):
    s = int(settings['map_size']/settings['graph_scale'])
    graph_x = ones(shape=(s**2,2))
    for i in range(s):
        graph_x[i*s:(i+1)*s,0] = [i*settings['graph_scale'] for j in range(s)]
        graph_x[i*s:(i+1)*s,1] = [j*settings['graph_scale'] for j in range(s)]
    print('Start drawing')
    phi = get_phi(graph_x)
    print('phi',phi.shape)
    graph_t = numpy.dot(phi,w)
    print('graph_t',graph_t.shape)
    return numpy.array(graph_x),numpy.array(graph_t)
