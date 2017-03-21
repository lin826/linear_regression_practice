# Reference:
# http://aimotion.blogspot.tw/2011/10/machine-learning-with-python-linear.html
# https://www.tutorialspoint.com/python/python_command_line_arguments.htm

import sys
from approach import *

# Approach method
APPR = Test_approach

# Parameters settings
settings = {"map_size":1081,"dim":2,"data_size":10000,
    "batch_size":1,"iter":10,"eta":0.8,"k_folder":0,
    "basis_size":1024,"sigma":30, "lamb":0.05,
    "m0":0,"alpha":2,"beta":25,"graph_scale":2,
    "x_train":"data/X_train.csv","t_train":"data/T_train.csv"}

# Update the parameters
def main_opt(opt, data):
    if(opt == "x_train" or opt == "t_train"):
        settings[opt] = data
    elif(opt == "approach"):
        set_approach(data)
    elif(opt == "eta" or opt == "lamb"):
        settings[opt] = float(data)
    else:
        settings[opt] = int(data)

# Choose the approach to run
def set_approach(data):
    global APPR
    if(data == "ML"):
        APPR = ML_approach
    elif(data == "MAP"):
        APPR = MAP_approach
    elif(data == "Bayesian"):
        APPR = Bayesian_approach
    elif(data == "Test"):
        APPR = Test_approach
    else:
        print("No approach:",data)


# Main() where the start of program
if __name__ == "__main__":
    arg = sys.argv[1:]
    for i in range(len(arg)-1):
        if(arg[i].startswith("--")):
            opt = arg[i][2:]
            main_opt(opt,arg[i+1])
    model_init(settings)
    for k in range(settings['k_folder']+1):
        if model_setting(settings,k+1) < 0:
            break
        set_Gausian_basis()
        weights = APPR()
    print(weights)
    cal_MSE(weights)
    draw(weights,show_2d_gragh)
