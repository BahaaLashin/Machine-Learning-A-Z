import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()
data_load = pd.DataFrame(data['data'],columns=data['feature_names'])

X = data_load
y = data['target']

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense

print(x_train.shape)
classifier = Sequential()
classifier.add(Dense(output_dim=15,input_dim=30,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=7,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))
classifier.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

ypred = classifier.predict(x_test)
ypred = (ypred > .5)
from sklearn.metrics import confusion_matrix,classification_report
print('confusion matrix : ',confusion_matrix(ypred,y_test))
print('classification report : ',classification_report(ypred,y_test))