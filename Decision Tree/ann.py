import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df = pd.read_csv('kyphosis.csv')
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

le = LabelEncoder()
y = le.fit_transform(y)

x_train , x_test ,y_train , y_test = train_test_split(X,y,test_size=.2)

from keras.models import Sequential
from keras.layers import Dense
print(x_train.shape)
classifier = Sequential()
classifier.add(Dense(input_dim=3,output_dim=2,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))
classifier.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
classifier.fit(x_train,y_train,batch_size=40,nb_epoch=100)
ypred = classifier.predict(x_test)
ypred = (ypred > .5)
from sklearn.metrics import confusion_matrix,classification_report
print('confusion matrix : ',confusion_matrix(ypred,y_test))
print('classification report : ',classification_report(ypred,y_test))

print(ypred,y_test)