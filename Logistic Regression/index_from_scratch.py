import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
age and cancer 
'''
dataset = np.array([[1,0],[2,0],[3,0],[4,0],[5,0],[6,1],[7,0],[8,1],[9,1],[10,1],[11,1],[12,1],[13,1],[14,1],[15,1],[16,1]])


def get_slope(x,y):
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(np.multiply(x,y))
    sum_x2 = np.sum(np.power(x,2))
    n = len(x)
    slope = (n*sum_xy - sum_x*sum_y)/(n*sum_x2 - sum_x**2)
    return slope

def get_intercept(x,y):
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(np.multiply(x, y))
    sum_x2 = np.sum(np.power(x, 2))
    n = len(x)
    intercept = (sum_y*sum_x2-sum_x*sum_xy)/(n*sum_x2-sum_x**2)
    return intercept


def get_sigmoid(x):
    return 1/(1+np.exp(-x))


X = dataset[:,:1]
y = dataset[:,1]

slope = get_slope(X,y)
intercept = get_intercept(X,y)
data_test = slope*np.arange(1,17)+intercept
sigmoid = get_sigmoid(data_test)

plt.plot(np.arange(1,17),sigmoid,color='red')
plt.scatter(X,y)
plt.show()