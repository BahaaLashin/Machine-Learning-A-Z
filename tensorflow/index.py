import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler

df = pd.read_csv('Churn_Modelling.csv')

x= df.iloc[:,3:13].values
y= df.iloc[:,13].values


label1 = LabelEncoder()
x[:,2] = label1.fit_transform(x[:,2])
label2 = LabelEncoder()
x[:,1] = label2.fit_transform(x[:,1])

oneHotEnc = OneHotEncoder(categorical_features=[1])
x = oneHotEnc.fit_transform(x).toarray()
x = x[:,1:]

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=.3)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)
classifier = Sequential()

classifier.add(Dense(output_dim=6,activation='relu',input_dim=11,init='uniform'))
classifier.add(Dense(output_dim=6,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > .5)
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_pred, y_test)
print(conf)