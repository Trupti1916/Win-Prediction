import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
#-----------------------------------------------------------------------------
dataset= pd.read_excel('Win_Prediction_Data.xlsx')

df = dataset.copy()
print(df)
print(df.info())
print("Shape -",df.shape)
print("="*70)

#find missing data, you will get True/False and Count
print("Missing Data- \n",df.isnull().any())
print("="*70)
print("Count Of Missing Data-\n",df.isnull().sum())

#we need to fill the missing values, in order to do so, we will check which category is used more number of times, 
# Since it's categorical data we will use Mode
Client_Category = df['Client Category'].value_counts()
print(Client_Category)
#Others Category is used maximum number of times, so we can use that to fill the missing values
df['Client Category'] = df['Client Category'].fillna('Others')
#-----------------------------------------------------------------------------

#Summary Of Categorical Variable
sumcat = df.describe(include="O")
print(sumcat)
#Summary of Float Variable
sumfloat = df.describe()
print(sumfloat)
#-----------------------------------------------------------------------------
#purpose of doing this is to find the top client Category, Solution, Managers, VP's

Client_Category_index = df['Client Category'].value_counts().index
print(Client_Category_index)

#Solution Type
Solution_Type = df['Solution Type'].value_counts()
print(Solution_Type)
Solution_Type_index = df['Solution Type'].value_counts().index
print(Client_Category_index)

#Sector
Sector = df['Sector'].value_counts()
print(Sector)
Sector_index = df['Sector'].value_counts().index
print(Sector_index)

#Location
Location = df['Location'].value_counts()
print(Location)
Location_index = df['Location'].value_counts().index
print(Location_index)

#VP Name
VP_Name = df['VP Name'].value_counts()
print(VP_Name)
VP_Name_index = df['VP Name'].value_counts().index
print(VP_Name_index)

#Manager_Name
Manager_Name = df['Manager Name'].value_counts()
Manager_Name_index = df['Manager Name'].value_counts().index

Deal_Status_Code = df['Deal Status Code'].value_counts()
print(Deal_Status_Code)
Deal_Status_CodeIndex = df['Manager Name'].value_counts().index

#checking the relation between different variables
rel_client_cat = df[['Client Category','Deal Status Code']].groupby(['Client Category',
'Deal Status Code']).size().reset_index().rename(columns = {0: 'Deal Status Count'})
print('='*70)
print(rel_client_cat)
print(pd.pivot_table(df, index='Deal Status Code', columns='Client Category', values='Deal Cost'))

Solution_Deal = df[['Solution Type', 'Deal Status Code']].groupby(['Solution Type', 
'Deal Status Code']).size().reset_index().rename(columns={0:'Deal Status Count'})
print('='*70)
print(Solution_Deal)
print(pd.pivot_table(df, index='Deal Status Code', columns= 'Solution Type', values = 'Deal Cost'))

#Based on Date, if we want to identify in which year sales were max, which quarter the sales were up etc
df['DealDate_year'] = df['Deal Date'].dt.year # this will give Year in the column
print('='*70)
print(df.head())
print('='*70)

df['DealDate_Month'] = df['Deal Date'].dt.month
print(df.head())

df['DealDate_Quarter'] = df['Deal Date'].dt.quarter
print('='*70)
print(df.columns)
df.pivot(columns=['DealDate_Quarter'], values=['Deal Cost']).plot.hist()

df = df.drop(['Deal Date'], axis=1)
print(df.head(5))

#Label Encoder
df['Client Category'] = df['Client Category'].astype('category')
df['Client Category'] = df['Client Category'].cat.codes 

df['Solution Type'] = df['Solution Type'].astype('category')
df['Solution Type']= df['Solution Type'].cat.codes

df['Sector'] = df['Sector'].astype('category')
df['Sector']= df['Sector'].cat.codes

df['Location'] = df['Location'].astype('category')
df['Location']= df['Location'].cat.codes

df['VP Name'] = df['VP Name'].astype('category')
df['VP Name']= df['VP Name'].cat.codes

df['Manager Name'] = df['Manager Name'].astype('category')
df['Manager Name']= df['Manager Name'].cat.codes

df['Deal Status Code'] = df['Deal Status Code'].astype('category')
df['Deal Status Code']= df['Deal Status Code'].cat.codes
print(df.head(5))
#================================================================================================

#Spliting the data
X = df.drop(['Deal Status Code'], axis=1)
X = X.iloc[:,0:7]
print(X.head(5))
y = df['Deal Status Code']

#Scaling Deal Cost 
sc = StandardScaler()
X1 = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=101, test_size=0.25)
#=======================================================================================================================

#KNN CLassifier
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
y_pred_train_knn = knn.predict(X_train)
y_pred_test_knn = knn.predict(X_test)

print("confusion_matrix",confusion_matrix(y_test, y_pred_test_knn))    
print("classification_report", classification_report(y_test, y_pred_test_knn))
print("Train accuracy_score ",accuracy_score(y_train, y_pred_train_knn))
print("Test accuracy_score ",accuracy_score(y_test, y_pred_test_knn))
#Train accuracy_score  0.7769383697813121
#Test accuracy_score  0.6637519872813991

#Cross Validation
acc_test_knn = cross_val_score(knn, X_test, y_test, cv=20)
print('-'*50)
print("Test Accuracy - ",acc_test_knn)
#Train accuracy_score  0.7769383697813121
#Test accuracy_score  0.73015873
#==================================================================================================================

#Naive Bayes 
nb = BernoulliNB()
nb.fit(X_train, y_train)
y_pred_train_nb = nb.predict(X_train)
y_pred_test_nb = nb.predict(X_test)

print(confusion_matrix(y_train, y_pred_train_nb))
print(confusion_matrix(y_test, y_pred_test_nb))
print('-'*50)
print(accuracy_score(y_train, y_pred_train_nb))
print(accuracy_score(y_test, y_pred_test_nb))

cross_val_train_nb = cross_val_score(nb, X_train, y_train, cv=10)
cross_val_test_nb = cross_val_score(nb, X_test, y_test, cv=10 )
print(cross_val_train_nb)
print(cross_val_test_nb)
#Test Accuracy = 0.6198807157057654
#Test Accuracy = 0.6482511923688394
#==================================================================================================================
