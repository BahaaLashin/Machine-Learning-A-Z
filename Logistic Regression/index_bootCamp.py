import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
data_set = pd.read_csv('titanic_train.csv')

data_set = data_set.dropna()

X = data_set.drop(['Name','Sex','Ticket','Cabin','Embarked'],axis=1)
y = data_set['Embarked']
y = y.map({'S':1,'C':0})

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=.2,random_state=101)


lr = LogisticRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)



