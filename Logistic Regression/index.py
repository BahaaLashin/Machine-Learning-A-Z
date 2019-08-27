import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
'''
age and cancer 
'''
dataset = np.array([[1,0],[2,0],[3,0],[4,0],[5,0],[6,1],[7,0],[8,1],[9,1],[10,1],[11,1],[12,1],[13,1],[14,1],[15,1],[16,1]])
X = dataset[:,:1]
y = dataset[:,1]
lr = LogisticRegression()
lr.fit(X,y)

slope = lr.coef_
intercept = lr.intercept_
data_test = lr.predict(np.arange(1,17).reshape(16,1))
plt.plot(np.arange(1,17),data_test,color='red')
plt.scatter(X,y)
plt.show()