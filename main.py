import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# %matplotlib inline
from matplotlib import pyplot as plt

train_df = pd.read_csv('howpop_train.csv')
test_df  = pd.read_csv('howpop_test.csv')

train_df.head(1).T

train_df.shape, test_df.shape

train_df['published'].apply(lambda ts: pd.to_datetime(ts).value).plot();

train_df.corr()

[abs(i) for i in np.array(train_df.corr()).flatten()
 if (i > 0.9) & (i < 1)]

train_df.head()

train_df['published'] = pd.to_datetime(train_df['published'], yearfirst=True)
train_df['year'] = [d.year for d in train_df.published]

train_df['year'].value_counts()

features = ['author', 'flow', 'domain','title']
train_size = int(0.7 * train_df.shape[0])

X, y = train_df.loc[:, features],  train_df['favs_lognorm']

X_test = test_df.loc[:, features]

X_train, X_valid = X.iloc[:train_size, :], X.iloc[train_size:,:]

y_train, y_valid = y.iloc[:train_size], y.iloc[train_size:]

vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, ngram_range=(1, 3))

X_train_tfidf = vectorizer.fit_transform(X_train['title'])

X_valid_tfidf = vectorizer.transform(X_valid['title'])

X_test_tfidf = vectorizer.transform(X_test['title'])

X_train_tfidf.shape

word = "python"
word_index = vectorizer.vocabulary_[word]
print(word_index)

vectorizer = TfidfVectorizer(analyzer='char')

X_train_vectorized = vectorizer.fit_transform(X_train['title'])
X_valid_vectorized = vectorizer.transform(X_valid['title'])
X_test_vectorized = vectorizer.transform(X_test['title'])

X_train_vectorized.shape

feats = ['author', 'flow', 'domain']

vectorizer_feats = DictVectorizer()

X_train_feats = vectorizer_feats.fit_transform(X_train.loc[:, feats].fillna('-').T.to_dict().values())
X_valid_feats = vectorizer_feats.transform(X_valid.loc[:, feats].fillna('-').T.to_dict().values())
X_test_feats = vectorizer_feats.transform(X_test.loc[:, feats].fillna('-').T.to_dict().values())

X_train_feats.shape

X_train_new = scipy.sparse.hstack([X_train_tfidf, X_train_feats, X_train_vectorized])
X_valid_new = scipy.sparse.hstack([X_valid_tfidf, X_valid_feats, X_valid_vectorized])
X_test_new =  scipy.sparse.hstack([X_test_tfidf, X_test_feats, X_test_vectorized])

# %%time
model1 = Ridge(alpha=0.1, random_state=1)
model1.fit(X_train_new, y_train)

train_preds1 = model1.predict(X_train_new)
valid_preds1 = model1.predict(X_valid_new)

print('Error train',mean_squared_error(y_train, train_preds1))
print('Error valid',mean_squared_error(y_valid, valid_preds1))

# %%time
model2 = Ridge(alpha=1.0,random_state=1)
model2.fit(X_train_new, y_train)

train_preds2 = model2.predict(X_train_new)
valid_preds2 = model2.predict(X_valid_new)

print('Error train', mean_squared_error(y_train, train_preds2))
print('Error valid', mean_squared_error(y_valid, valid_preds2))