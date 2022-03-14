#importing libraries
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import nltk
import re
import string
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score





#Importing the Data
data = pd.read_csv('clean_final_df.csv' , lineterminator='\n')  






#preprocessing the Data

import re

def remove_usernames_links(tweet):
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('http[^\s]+','',tweet)
    return tweet


def remove_emoji(tweet):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweet)



#Splitting & vectorizing the Data



X_train , X_test , y_train , y_test = train_test_split(data['tweet'] , data['dialect'] , test_size = 0.30 , random_state = 53)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


tf_idf = TfidfVectorizer()
X_train_tf = tf_idf.fit_transform(X_train)
X_train_tf = tf_idf.transform(X_train)
print("n_samples: %d, n_features: %d" % X_train_tf.shape)
X_test_tf = tf_idf.transform(X_test)
print("n_samples: %d, n_features: %d" % X_test_tf.shape)



#logistic Regression model & the prediction

LR_clf = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
model=LR_clf.fit(X_train_tf,y_train)


y_pred_lR = LR_clf.predict(X_test_tf)
print(metrics.classification_report(y_test, y_pred_lR, target_names=['IQ', 'LY', 'QA', 'PL'
, 'SY', 'TN', 'JO', 'MA', 'SA', 'YE', 'DZ','EG', 'LB', 'KW', 'OM', 'SD', 'AE', 'BH']))

test_acc = LR_clf.score(X_test_tf , y_test)
print("test accuracy is :" ,test_acc)
train_acc = LR_clf.score(X_train_tf , y_train)
print("train accuracy is : " ,train_acc)


#prediction Function 
