import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
data_load = pd.DataFrame(data['data'],columns=data['feature_names'])

X = data_load
y = data['target']

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.3)

cl = SVC()
cl.fit(x_train,y_train)
y_pred = cl.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
print('SVM')
print(confusion_matrix(y_test,y_pred))
print('\n')
print(classification_report(y_pred,y_test))

plt.scatter(range(len(y_pred)),y_pred,color='red',s=100)
plt.scatter(range(len(y_pred)),y_test,color='blue')
plt.show()