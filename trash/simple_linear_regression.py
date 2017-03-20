import numpy as np
from numpy import zeros

regularization = 0
sigma = 30

alpha = 1.0
beta = 1.0

lamda = alpha/beta

# Gausian/Normal distribution
def set_regularization(test,theta):
	for e in theta:
		regularization += e**2
	regularization *= lamda

def get_sum_x_poly(test):
	sum_x_poly = []
	for i in range(len(test)):
		print(i)
		sum_x_poly.append(test**i)
	return np.matrix(sum_x_poly)

def get_inverse_S(predict):
	return alpha*np.identity(len(predict))+beta*(predict.transpose().dot(predict))

def get_S(predict):
	return np.linalg.inv(get_inverse_S(predict))

# mean_sigma = [[mean_x, mean_y, sigma=50]...]
mean_sigma = []
def set_Gausian(it, y, theta,MAP_SIZE):
	l = int(len(theta)**0.5)
	for i in range(l):
		for j in range(l):
			x = (i+1)*MAP_SIZE/(l+1)
			y = (j+1)*MAP_SIZE/(l+1)
			mean_sigma.append([x,y,sigma])
	# print(mean_sigma[:len(theta)])

def get_gradient(X,y,p,j):
	x = (X[:,0] - mean_sigma[j][0])**2 + (X[:,1] - mean_sigma[j][1])**2
	x = x**-0.5
	return ((p - y) * x).sum() / x.sum()

def simple_Gausian(test,theta):
	result = []
	for row in test:
		a = 0
		for i in range(len(theta)):
			# fi = 100 / (mean_sigma[i][2] * np.sqrt(2 * np.pi)) * np.exp( - ((row[1] - mean_sigma[i][0])**2+(row[2] - mean_sigma[i][1])**2) / (2 * mean_sigma[i][2]**2) )
			fi = 100 * np.exp( - ((row[0] - mean_sigma[i][0])**2+(row[1] - mean_sigma[i][1])**2) / (2 * mean_sigma[i][2]**2) )
			# print(row[1],row[2],i, theta[i][0]*fi)
			a += theta[i][0]*fi
		result.append([a])
	# print(np.array(result))
	return np.array(result)

def MAP_Gausian(test,theta,lamda):
	result = []
	for row in test:
		a = 0
		for i in range(len(theta)):
			# fi = 100 / (mean_sigma[i][2] * np.sqrt(2 * np.pi)) * np.exp( - ((row[1] - mean_sigma[i][0])**2+(row[2] - mean_sigma[i][1])**2) / (2 * mean_sigma[i][2]**2) )
			fi = 100 * np.exp( - ((row[0] - mean_sigma[i][0])**2+(row[1] - mean_sigma[i][1])**2) / (2 * mean_sigma[i][2]**2) )
			# print(row[1],row[2],i, theta[i][0]*fi)
			a += theta[i][0]*fi
		result.append([a+regularization])
	# print(np.array(result))
	return np.array(result)


# Change algorithm to use here!!!
def get_predict_value(test,theta):
	# return MAP_Gausian(test,theta,lamda)
    return simple_Gausian(test,theta)

def Bayesian_mean(S,predict,y):
	return beta * S.dot(predict.transpose().dot(y))

def get_theta_Bayssian(it,y,theta):
	predict = get_predict_value(it,theta)
	S = get_S(predict)
	return Bayesian_mean(S,predict,y)
