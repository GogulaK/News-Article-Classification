import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reading the given data files
test = pd.read_csv("./Data/test_v2.csv")
train = pd.read_csv("./Data/train_v2.csv")

# TfidfVectorization and NB Classification without pipeline
'''titlev = TfidfVectorizer(stop_words='english')
title = titlev.fit_transform(train['title'])

nb = MultinomialNB()
nb.fit(title, train['category'])
cat_pred = nb.predict(titlev.transform(test['title']))'''

# TfidfVectorization and NB Classification using pipeline
'''pipe1 = Pipeline([
  ('vectorize', TfidfVectorizer(stop_words='english')),
  ('classify', MultinomialNB())
  ])
pipe1.fit(train['title'], train['category'])
cat_pred = pipe1.predict(test['title'])'''

# CountVectorization and NB Classification using pipeline without stop words
'''pipe2 = Pipeline([
    ('vectorize', CountVectorizer()),
    ('transform', TfidfTransformer()),
    ('classify', MultinomialNB())
  ])
pipe2.fit(train['title'], train['category'])
cat_pred = pipe2.predict(test['title'])'''

# TfidfVectorization and SVM Classification using pipeline
'''pipe3 = Pipeline([
    ('vectorize', TfidfVectorizer(stop_words='english')),
    ('classify', svm.SVC(kernel='linear'))
  ])
pipe3.fit(train['title'], train['category'])
cat_pred = pipe3.predict(test['title'])'''

# CountVectorization and SGDClassifier using pipeline
'''pipe4 = Pipeline([('vectorize', CountVectorizer(stop_words='english')),
                  ('transform', TfidfTransformer()),
                  ('classify', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
                  ])
pipe4.fit(train['title'], train['category'])
cat_pred = pipe4.predict(test['title'])'''

# TfidfVectorization and SGDClassifier using pipeline
'''pipe5 = Pipeline([('vectorize', TfidfVectorizer(stop_words='english')),
                  ('classify', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
                  ])
pipe5.fit(train['title'], train['category'])
cat_pred = pipe5.predict(test['title'])'''

# CountVectorization and SGDClassifier using pipeline along with GridSearchCV - alpha,iter modified

# Title field pre processing
# t = train.iloc[0]['title']
'''stemmer = PorterStemmer()
# lemmatizer=WordNetLemmatizer()
for index, row in train.iterrows():
    t = row['title']
    t = t.lower()
    t = t.translate(str.maketrans('', '', string.punctuation))
    t_token = word_tokenize(t)
    t_temp = " "
    for word in t_token:
        t_temp += stemmer.stem(word) + " "
    t = t_temp.strip()
    train.at[index, 'title'] = t'''


# Train test split
X_train, X_test, y_train, y_test = train_test_split(train['title'], train['category'], test_size=0.2, random_state=15)
parameters = {'vectorize__ngram_range': [(1, 1), (1, 2)],
              'transform__use_idf': (True, False),
              'classify__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
              }
pipe6 = Pipeline([('vectorize', CountVectorizer(stop_words='english')),
                  ('transform', TfidfTransformer()),
                  ('classify', SGDClassifier(loss='hinge', penalty='l2', n_iter=50, random_state=42))
                ])
pipe6 = GridSearchCV(pipe6, parameters, n_jobs=-1, cv=3)
# pipe6.fit(train['title'], train['category'])
pipe6.fit(X_train, y_train)
print(pipe6.best_params_)
t_cat_pred = pipe6.predict(X_test)
print("Score:", accuracy_score(t_cat_pred, y_test))


cat_pred = pipe6.predict(test['title'])
# Writing the predictions into the result file
test['category'] = cat_pred
test = test.drop(['title', 'url', 'publisher', 'hostname', 'timestamp'], axis=1)
test.to_csv("./Data/result.csv")
