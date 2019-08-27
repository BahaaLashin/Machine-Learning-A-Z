'''
principal components analysis
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
data_load = pd.DataFrame(data['data'],columns=data['feature_names'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_load)

model = PCA(n_components=2)
data_PCA =  model.fit_transform(data_scaled)
plt.scatter(data_PCA[:,0],data_PCA[:,1],c=data['target'])
plt.show()