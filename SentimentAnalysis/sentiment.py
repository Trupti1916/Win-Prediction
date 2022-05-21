from itertools import count
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

train_ds = pd.read_csv('sentiment', delimiter='\t')
print(train_ds.tail(5))

#in Sentiment column we have 2 clasess (0 & 1), 0 - Negative Sentiment and 1 - Positive sentiment

#comparing and doing filtering the positive sentiments only
print(train_ds[train_ds['sentiment']==1][0:10])

print(train_ds.info())

#Plot graph to check how many 1 and 0, Data visualization
plt.figure(figsize=(8,6))
#Create count plot
ax = sns.countplot(x='sentiment', data = train_ds)
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x() + 0.1 , p.get_height() + 50))
    plt.show()

#given count for both the classes
review_vol = train_ds["sentiment"].value_counts()
print(review_vol)

print("="*50)
#to check the percentage of positive and negative
print((review_vol[0]/train_ds.shape[0])*100)
print("="*50)
print((review_vol[1]/train_ds.shape[0])*100)

print("-"*50)
#Removing Stop_Words
stop_words = text.ENGLISH_STOP_WORDS
print("Stop Words \n", len(stop_words))
print("="*50)
#Adding more stop word to the list
stop_words = text.ENGLISH_STOP_WORDS.union(['ML', 'happy','code', 'movies','job' ])

#Bag Of Words
cv = CountVectorizer(stop_words=stop_words, max_features=1000)
feature_vector = cv.fit(train_ds.text)

#get feature name
print("-"*50)
feature = feature_vector.get_feature_names()
print("Total Number of Feature ",len(feature))
print("-"*50)
print(random.sample(feature,20))

train_ds_feature = cv.transform(train_ds.text)
#print(train_ds_feature)
train_ds_df = pd.DataFrame(train_ds_feature.toarray())

#Counting the frequency of words..
feature_count = np.sum(train_ds_feature.toarray(), axis=0)
feature_count_df = pd.DataFrame(dict(features = feature,    count = feature_count))
print(feature_count_df.head(10))

sort_feature =(feature_count_df.sort_values('count', ascending=False)[0:25])
print("="*50)
print(sort_feature)

#Converting the vector matrix into dataframe
train_df = pd.DataFrame(train_ds_feature.todense())
train_df.columns = feature
train_ds_df['sentiment'] = train_ds.sentiment
train_ds_df.head()

#Splitting the Data set into dependent and independent. 
Xtrain, Xtest, ytrain, ytest = train_test_split(train_ds_feature, train_ds.sentiment, test_size=0.30)
print("="*50)
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)

#Building Machine Learning Model
#1. Naive Bayes Theorem
print('Naive Bayes Theorem')
nb_classifier = BernoulliNB()
nb_classifier.fit(Xtrain.toarray(), ytrain)

#prediction
test_pred = nb_classifier.predict(Xtest.toarray())

cm = confusion_matrix(ytest,test_pred)
print(cm)
print("classification_report : --\n",classification_report(ytest,test_pred))
print("Accuracy Score : --",accuracy_score(ytest,test_pred))
sns.heatmap(cm, annot=True, fmt='.2f')
print("="*50)

COLUNM_NAME = ['Process', 'Model Name', 'F1-Score', 'Range of F1-Score', 'Std Deviation of F1-Screa']
df_Model_Selection =  pd.DataFrame(columns = COLUNM_NAME)
print(df_Model_Selection)

#Cross Validation
def stratified_k_fold_validation(model_obj, model_name, process, n_splits,x,y):
    global df_Model_Selection
    skf = StratifiedKFold(n_splits)

    weighted_f1_score =[]
    print(skf.split(x,y))

    for train_index, val_index in skf.split(x,y):
        x_train, x_test = x[train_index], x[val_index]
        y_train, y_test = y[train_index], y[val_index]

        model_obj.fit(x_train, y_train)
        test_ds_predicted = model_obj.predict(x_test)
        weighted_f1_score.append(round(f1_score(y_test, test_ds_predicted, average='weighted'),2))
    std_dev = np.std(weighted_f1_score, ddof=1)        
    range_f1_score = "{}-{}".format(min(weighted_f1_score), max(weighted_f1_score))

    df_Model_Selection = pd.concat([df_Model_Selection, 
                                    pd.DataFrame([[process, model_name, sorted(weighted_f1_score),
                                    range_f1_score,std_dev]],columns=COLUNM_NAME)])

model_obj= nb_classifier
model_name = 'Naive Bayes Classifier'
process = 'BOW with NLTK Lemmatization'
n_splits = 5
x= train_ds_feature.toarray()
y= train_ds.sentiment
stratified_k_fold_validation(model_obj, model_name, process, n_splits,x,y)
print(df_Model_Selection)

#2.Logistic Regression
print('Logistic Regression :--')
logit = LogisticRegression()
logit.fit(Xtrain.toarray(), ytrain)

test_predict = logit.predict(Xtest.toarray())
cm1 = metrics.confusion_matrix(ytest, test_predict)
print(cm1)
acc = metrics.accuracy_score(ytest, test_predict)
acc = metrics.accuracy_score(ytest, test_predict)
print(acc)

model_obj= logit
model_name = 'Logistic Regression'
process = 'BOW with NLTK Lemmatization'
n_splits = 5
x= train_ds_feature.toarray()
y= train_ds.sentiment
stratified_k_fold_validation(model_obj, model_name, process, n_splits,x,y)
print('-'*70)
print(df_Model_Selection)
print('-'*70)
