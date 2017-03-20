# Reference:
# http://aimotion.blogspot.tw/2011/10/machine-learning-with-python-linear.html
# https://www.tutorialspoint.com/python/python_command_line_arguments.htm

import sys
from approach import *

APPR = Test_approach
settings = {'map_size':1081,'dim':2,'iter':10,'eta':0.005,
    'basis_size':1024,'sigma':30, 'lambda':0.1,
    'x_train':'data/X_train.csv','t_train':'data/T_train.csv'}

def main_opt(opt, data):
    if(opt == 'x_train' or 't_train'):
        settings[opt] = data
    elif(opt == 'approach'):
        set_approach(data)
    elif(opt == 'eta' or 'lambda'):
        settings[opt] = float(data)
    else:
        settings[opt] = int(data)


def set_approach(data):
    if(data == 'ML'):
        APPR = ML_approach
    elif(data == 'MAP'):
        APPR = MAP_approach
    elif(data == 'Baysian'):
        APPR = Baysian_approach
    elif(data == 'Test'):
        APPR = Test_approach
    else:
        print('No approach:',data)

if __name__ == "__main__":
    arg = sys.argv[1:]
    for i in range(len(arg)-1):
        if(arg[i].startswith('--')):
            opt = arg[i][2:]
            main_opt(opt,arg[i+1])
    model_setting(settings)
    APPR()
