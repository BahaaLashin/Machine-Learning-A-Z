import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

columns_name = ['user_id','item_id','rate','timestamp']
data = pd.read_csv('u.data',sep='\t',names=columns_name)
movies = pd.read_csv('Movie_Id_Titles')
df = pd.merge(data,movies,on='item_id')
ratings = df.groupby('title')['rate'].count()
# print(ratings.head())
movies_table = pd.pivot_table(data=df,index='user_id',columns='title',values='rate')

'''closest movie to Toy Story'''

toy_story_rate = movies_table['Batman Forever (1995)']
closest_movies_to_toy = pd.DataFrame(movies_table.corrwith(toy_story_rate),columns=['cor'])

closest_movies_to_toy.dropna(inplace=True)
# closest_movies_to_toy = closest_movies_to_toy.sort_values(1)
cro_toy = closest_movies_to_toy.join(ratings,on='title')
print(cro_toy[cro_toy['rate'] > 100].sort_values('cor',ascending=False))
