import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_slope(x,y):
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(np.multiply(x,y))
    slope = (len(x)*sum_xy-sum_x*sum_y)/(len(x)*np.sum(x**2) - sum_x**2)
    return slope

def get_intercept(x,y):
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(np.multiply(x, y))
    sum_x2 = np.sum(x**2)
    intercept = (sum_y*sum_x2-sum_x*sum_xy)/(len(x)*sum_x2 - sum_x**2)
    return intercept
dataset =np.array([[1,2],[2,3],[3,5],[4,4],[5,4],[6,5],[7,7],[8,6],[9,8],[10,9]])

X = dataset[:,0]
y = dataset[:,1]
slope = get_slope(X,y)
intercept = get_intercept(X,y)
line = slope * np.arange(0,10) + intercept
plt.plot(range(0,10), line)
plt.scatter(X,y)
plt.show()

