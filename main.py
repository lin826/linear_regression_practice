# Reference:
# http://aimotion.blogspot.tw/2011/10/machine-learning-with-python-linear.html
# https://www.tutorialspoint.com/python/python_command_line_arguments.htm

import sys
from approach import *

# Approach method
APPR = Test_approach

# Parameters settings
settings = {"map_size":1081,"dim":2,"data_size":32000,
    "batch_size":1,"iter":100,"eta":0.5,"k_folder":0,
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

def try_test(x_file,w):
    test_x = get_loc_data(x_file)
    graph_x, graph_t = get_graph_data(w)
    result = []
    with open('data/'+str(APPR)+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar="'", quoting=csv.QUOTE_ALL)
        for i in range(len(test_x)):
            for j in range(len(graph_x)):
                if graph_x[j][0] == test_x[i][0] and graph_x[j][1] == test_x[i][1] :
                    writer.writerow([graph_t[j]])
                    print(test_x[i],graph_t[j])

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
    # print(weights)
    cal_MSE(weights)
    # draw(weights,show_2d_gragh)
    # try_test('data/X_test.csv',weights)
