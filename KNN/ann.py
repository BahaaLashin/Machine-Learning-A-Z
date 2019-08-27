import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
df = pd.read_csv('Classified Data.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

classifier = Sequential()
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=.2)
classifier.add(Dense(input_dim=11,activation='relu',output_dim=6,init='uniform'))
classifier.add(Dense(activation='relu',output_dim=6,init='uniform'))
classifier.add(Dense(activation='sigmoid',output_dim=1,init='uniform'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

ypred = classifier.predict(x_test)
ypred = (ypred > .5)
from sklearn.metrics import confusion_matrix,classification_report
print('confusion matrix : ',confusion_matrix(ypred,y_test))
print('classification report : ',classification_report(ypred,y_test))
