import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('wine.data.csv')

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=.2)

cluster = KMeans(n_clusters=3)
cluster.fit(x,y)
centroids = cluster.cluster_centers_
print(centroids)
