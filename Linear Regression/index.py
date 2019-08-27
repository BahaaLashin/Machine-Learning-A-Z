import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset =np.array([[1,2],[2,3],[3,5],[4,4],[5,4],[6,5],[7,7],[8,6],[9,8],[10,9]])

X = dataset[:,:1]
y = dataset[:,1]

lr = LinearRegression()
lr.fit(X,y)

slope = lr.coef_
intercept = lr.intercept_

line = slope * np.arange(0,10) + intercept
plt.plot(range(0,10), line)
plt.scatter(X,y)
plt.show()