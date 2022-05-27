"""Problem - Your Organization puts in a lot of effort in bidding preparation with no indications whether it will be worth it. With multiple bid managers and SBU Heads willing to
work on every opportunity, it becomes difficult for the management to decide which bid should be given to which bid manager and SBU Head. You are hired to help
your organization identify the best bid manager-SBU Head combination who can convert an opportunity to win with the provided data points. Objective 1: Predictive
Analytics - Build a ML model to predict the probability of win/loss for bidding activities for a potential client. Objective 2: Prescriptive Analytics – Identify variable/s
that are most likely to help in converting an opportunity into a win."""

"""Objective -- Predictive Analytics - Build a ML model to predict the probability of win/loss for bidding activities for a potential client.
➢ Prescriptive Analytics – Identify variable/s that are most likely to help in converting an opportunity into a win.
➢ Recommending top 5 Head-Bid Manager.
➢ For every false prediction calculate the loss which the company will face."""
#-----------------------------------------------------------------------------
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
#-----------------------------------------------------------------------------
#error = xlrd.biffh.XLRDError: Excel xlsx file; not supported
#Solution - pip install xlrd==1.2.0
#pip install pandas --upgrade
#pip install openpyxl
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
#Intution - missing data is present in Client Category Column only, 
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
plt.show()
#we need to fill the missing values, in order to do so, we will check which category is used more number of times, 
# Since it's categorical data we will use Mode
Client_Category = df['Client Category'].value_counts()
print(Client_Category)
#Others Category is used maximum number of times, so we can use that to fill the missing values
df['Client Category'] = df['Client Category'].fillna('Others')
#sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='coolwarm')
#plt.show()
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
#Plotting the top 5 Category
#plt.pie(Client_Category[:8], labels=Client_Category_index[:8], autopct='%1.2f%%')
#plt.show()
#Client_Category[:20].plot(kind='barh')
#plt.show()

#Solution Type
Solution_Type = df['Solution Type'].value_counts()
print(Solution_Type)
Solution_Type_index = df['Solution Type'].value_counts().index
print(Client_Category_index)
#Plotting the top 5 Solutio Type
#plt.pie(Solution_Type[:5], labels = Solution_Type_index[:5], autopct='%1.2f%%')
#plt.show()
#Solution_Type[:10].plot(kind='barh')
#plt.show()

#Sector
Sector = df['Sector'].value_counts()
print(Sector)
Sector_index = df['Sector'].value_counts().index
print(Sector_index)
#Plotting the top 5 Solutio Type
#plt.pie(Sector[:5], labels = Sector_index[:5], autopct='%1.2f%%')
#plt.show()
#Solution_Type[:10].plot(kind='barh')
#plt.show()

#Location
Location = df['Location'].value_counts()
print(Location)
Location_index = df['Location'].value_counts().index
print(Location_index)
#Plotting the top 5 Solutio Type
#plt.pie(Location[:5], labels = Location_index[:5], autopct='%1.2f%%')
#plt.show()
#Location[:10].plot(kind='barh')
#plt.show()

#VP Name
VP_Name = df['VP Name'].value_counts()
print(VP_Name)
VP_Name_index = df['VP Name'].value_counts().index
print(VP_Name_index)
#Plotting the top 5 Solutio Type
#plt.pie(VP_Name[:5], labels = VP_Name_index[:5], autopct='%1.2f%%')
#plt.show()
#VP_Name[:10].plot(kind='barh')
#plt.show()

#Manager_Name
Manager_Name = df['Manager Name'].value_counts()
Manager_Name_index = df['Manager Name'].value_counts().index
#Plotting the top 5 Solutio Type
#plt.pie(Manager_Name[:5], labels = Manager_Name_index[:5], autopct='%1.2f%%')
#plt.show()
#
# Manager_Name[:10].plot(kind='barh')
#plt.show()

Deal_Status_Code = df['Deal Status Code'].value_counts()
print(Deal_Status_Code)
Deal_Status_CodeIndex = df['Manager Name'].value_counts().index
#plt.pie(Deal_Status_Code[:2], labels = Deal_Status_CodeIndex[:2], autopct='%1.2f%%')
#plt.show()

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
#df.pivot(columns=['DealDate_Quarter'], values=['Deal Cost']).plot.hist()
#sns.histplot(df.DealDate_Quarter, bins=4)
#plt.show()

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

#Logistic Regression
logit = LogisticRegression()
logit.fit(X_train, y_train)

y_pred_test = logit.predict(X_test)
y_pred_train = logit.predict(X_train)

print(confusion_matrix(y_train, y_pred_train))
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))
print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_test, y_pred_test))

#In building the mmodels, the major problem is overfitting (High Variance) and underfitting (High Bias)
#Accuracy scores is bad for train = 0.6196156394963552 and test =0.6482511923688394. We can see there is high Bias, 
#so we will use XGBoost, but before that we can try cross validation K -Fold method
acc_train = cross_val_score(logit, X_train, y_train, cv =20)
acc_test = cross_val_score(logit, X_test, y_test, cv=20)
print('-'*50)
print("Train Accuracy - ",acc_train)
print("Test Accuracy - ",acc_test)
#Training Accuracy = 0.6196156394963552
#Test Accuracy = 0.6482511923688394
#==========================================================================================

#XGBoost Classifier
xb_class = XGBClassifier()
xb_class.fit(X_train, y_train)
y_pred_xgb_train = xb_class.predict(X_train)
y_pred_xgb_test = xb_class.predict(X_test)

print(confusion_matrix(y_train, y_pred_xgb_train))
print(confusion_matrix(y_test, y_pred_xgb_test))
print('-'*50)
print(classification_report(y_train, y_pred_xgb_train))
print(classification_report(y_test, y_pred_xgb_test))
print('-'*50)
print(accuracy_score(y_train, y_pred_xgb_train))
print(accuracy_score(y_test, y_pred_xgb_test))

acc_test_xgb = cross_val_score(xb_class, X_test, y_test, cv=20)
print('-'*50)
print("Test Accuracy - ",acc_test_xgb)
print("XGBoost Test Accuracy",acc_test_xgb[10])

#Training Accuracy = 0.9271040424121935
#Test Accuracy = 0.8333333333333334
#================================================================================================================

#Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_train_rf = rf.predict(X_train)
y_pred_test_rf = rf.predict(X_test)
print("Train Accuracy",accuracy_score(y_train, y_pred_train_rf))
print("Test Accuracy",accuracy_score(y_test, y_pred_test_rf))
acc_test_rf = cross_val_score(rf, X_test, y_test, cv=20)
print('-'*50)

#Train Accuracy = 0.9965540092776674
#Test Accuracy = 0.8259141494435612