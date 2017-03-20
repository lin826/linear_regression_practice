import numpy
from draw_graph import *
from data_io import *
from basic_cal import *

def model_setting(s):
    global Train_x,Train_t,settings
    Train_x, Train_t = get_train_data(s['x_train'],s['t_train'])
    Train_x = numpy.array(Train_x[:400])
    Train_t = numpy.array(Train_t[:400])
    settings = s

# Test on ML now
def Test_approach():
    set_Gausian_mean(settings['basis_size'],settings['map_size'],grid_mean)
    set_Gausian_sigma(settings['sigma'])
    weights = gradient_descent(Train_x,Train_t, settings['eta'], settings['iter'])
    # print(weights.size,'=',len(weights),'*',len(weights[0]))
    graph_x,graph_t = get_graph_data(weights)
    show_2d_gragh(graph_x,graph_t,settings['map_size'])
    return 0

def get_graph_data(w):
    s = settings['map_size']
    graph_x = [[i,j] for j in range(s) for i in range(s)]
    print('Start get graph_t')
    graph_t = w.dot(get_phi(graph_x))
    return numpy.array(graph_x),numpy.array(graph_t)
