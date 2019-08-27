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
from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(x_train,y_train)


y_pred = model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))