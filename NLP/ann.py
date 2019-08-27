import nltk
from nltk.corpus import stopwords
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
def text_procession(text):
    sTh = string.punctuation
    stopword = stopwords.words('english')
    text = [c for c in text if c not in sTh]
    text = ''.join(text).split()
    text = [word for word in text if word.lower() not in stopword ]
    return text

df = pd.read_csv('SMSSpamCollection',sep='\t',names=['type','message'])
# df['message'] = df['message'].apply(text_procession)
count_vec = CountVectorizer(analyzer=text_procession).fit(df['message'])
# print(count_vec.vocabulary_)

data_trans = count_vec.transform(df['message'])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf =TfidfTransformer()
tfidf = tfidf.fit_transform(data_trans)

from sklearn.model_selection import train_test_split
x_train , x_test,y_train ,y_test = train_test_split(tfidf,df['type'],test_size=.2)

print(x_train.shape)
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(input_dim=11425,output_dim=100,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=50,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=25,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=10,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=5,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))
classifier.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

ypred = classifier.predict(x_test)
ypred = (ypred > .5)
from sklearn.metrics import confusion_matrix,classification_report
print('confusion matrix : ',confusion_matrix(ypred,y_test))
print('classification report : ',classification_report(ypred,y_test))
