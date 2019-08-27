import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Classified Data.csv')

X = df.drop('TARGET CLASS',axis=1)
y = df['TARGET CLASS']

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
for i in np.arange(1,100):
    cl = KNeighborsClassifier(n_neighbors=i)
    cl.fit(x_train,y_train)
    print('neighbor is',i,' ',r2_score(y_test,cl.predict(x_test)))

