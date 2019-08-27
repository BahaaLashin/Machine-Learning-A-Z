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


cl = DecisionTreeClassifier()
cl.fit(x_train,y_train)
y_pred = cl.predict(x_test)

from sklearn.metrics import confusion_matrix , r2_score , classification_report
print('Decision Tree')
print(r2_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators=10)
rfcl.fit(x_train,y_train)

rfcl_y_pred = rfcl.predict(x_test)

print('Random Forest Classifier')
print(r2_score(y_test,rfcl_y_pred))
print(confusion_matrix(y_test,rfcl_y_pred))
print(classification_report(y_test,rfcl_y_pred))