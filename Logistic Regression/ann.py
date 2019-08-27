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

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=.1,random_state=101)

from keras.models import Sequential
from keras.layers import Dense

cls = Sequential()
cls.add(Dense(input_dim=7,output_dim=4,activation='relu',init='uniform'))
cls.add(Dense(output_dim=3,activation='relu',init='uniform'))
cls.add(Dense(output_dim=2,activation='relu',init='uniform'))
cls.add(Dense(output_dim=1,activation='relu',init='uniform'))

cls.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
cls.fit(x_train,y_train,batch_size=50,nb_epoch=200)

ypred = cls.predict(x_test)
ypred = (ypred > .5)
from sklearn.metrics import confusion_matrix,classification_report
print('confusion matrix : ',confusion_matrix(ypred,y_test))
print('classification report : ',classification_report(ypred,y_test))
