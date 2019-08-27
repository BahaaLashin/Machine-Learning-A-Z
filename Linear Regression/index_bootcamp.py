import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataSet = pd.read_csv('USA_Housing.csv')
X = dataSet[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y = dataSet['Price']

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(X,y,train_size=.8,test_size=.2)

lr = LinearRegression()
lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)
from sklearn import metrics

mae = metrics.mean_absolute_error(y_test,y_predict)
mse = metrics.mean_squared_error(y_test,y_predict)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test,y_predict)

print('mae',mae)
print('mse',mse)
print('rmse',rmse)
print('r2',r2)


plt.scatter(range(len(y)),y)
plt.show()