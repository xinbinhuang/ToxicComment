#!/usr/bin/env python
# Train a Logistic regression model on tf-idf features 
# and output the probability prediction for each toxic class

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# load data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# combine text for preprocessing
train_text = train_df["comment_text"]
test_text = test_df["comment_text"]
all_text = pd.concat([train_text, test_text])

# transform comment into tfidf features
tfidf_vectorizer = TfidfVectorizer(encoding="unicode",
                                   analyzer="word",
                                   token_pattern="\w{1,}",
                                   stop_words="english",
                                   ngram_range=(1, 1),
                                   max_features=10000)
tfidf_vectorizer.fit(all_text)
train_features = tfidf_vectorizer.transform(train_text)
test_features = tfidf_vectorizer.transform(test_text)

# train the model
scores = []
y_pred = pd.DataFrame.from_dict({id: test_df["id"]})
for class_name in class_names:
    train_target = train_df[class_name]
    classifier = LogisticRegression(solver="sag")

    cv_score = np.mean(cross_val_score(classifier, train_features,
                                       train_target, cv=3, scoring="roc_auc"))
    scores.append(cv_score)
    print("Average CV score for {} : {}".format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    y_pred[class_name] = classifier.predict_proba(test_features)[:, 1]
    
print("Total average CV score for all classes : {}".format(np.mean(scores)))

