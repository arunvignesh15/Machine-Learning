# Simple Linear Regression

#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

#Load data
dataset = pd.read_csv("LR_11.csv",header = 0)
print ("Data Shape: ", dataset.shape)
print ( dataset.head() )
print ( dataset.dtypes )

#Get all categorical variables and create dummies
obj = dataset.dtypes == np.object
print (obj)
dummydf = pd.DataFrame()

for i in dataset.columns[obj]:
    dummy = pd.get_dummies(dataset[i], drop_first=True)
    dummydf = pd.concat([dummydf, dummy], axis = 1)

print(dummydf)

#Merge the dummy and dataset
data = dataset
data = pd.concat([data,dummydf], axis = 1)
print ("head \n" , data.head())

obj1 = data.dtypes == np.object
print (obj1)
data = data.drop(data.columns[obj1], axis = 1)
print ("head after removal \n ", data.head())

#Declare the dependent variable and create your independent and dependent datasets
dep = 'House Price'
X = data.drop(dep, axis = 1)
y = data[dep]

#Scatter plots
seaborn.pairplot(data, kind='reg')

#Split into train and test
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split( X, y , test_size =0.2,random_state =5)

#Run model
import statsmodels.formula.api as sm
lm = sm.OLS(y_train ,X_train ).fit()
lm.summary()

#to check residual behaviour 
pred_train = lm.predict(X_train)
err_train = pred_train - y_train

print ("Error Prediction in Train: \n", err_train)

#Predict
pred_test = lm.predict(X_test)
err_test = pred_test - y_test

print ("Error Prediction in Test: \n", err_test)

#Actual vs predicted plot
plt.scatter(y_train,pred_train)
plt.xlabel('Y')
plt.ylabel('Pred')
plt.title('Main')

#Root Mean sq error
rmse = np.sqrt(np.mean((err_test))**2)
rmse

#Residual plot
plt.scatter(pred_train, err_train, c='b', s = 40, alpha = 0.5)
plt.scatter(pred_test,err_test, c="g", s=40)
plt.hlines(y=0, xmin=0, xmax=500)
plt.title('Residual plot - Train(blue), Test(green)')
plt.ylabel('Residuals')

#multicollinearity
cor = X.corr(method = 'pearson')
print (cor)

#Create a mask to diplay only lower triangle of the matrix
mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)]=True
seaborn.heatmap(cor,vmax=1,vmin=-1,cmap='RdYlGn_r',mask=mask)

dataset.head()

#Normal distribution
import statsmodels.api as sma
lm = sm.OLS(y,X).fit()
lm.summary()
datares = pd.concat([dataset, pd.Series(lm.resid, name ='resid'), pd.Series(lm.predict(), name = "Predict")], axis =1)

sma.qqplot(datares.resid)
plt.show()
print(lm.resid)

import scipy.stats as scipystats
import pylab
scipystats.probplot(datares.resid, dist='norm', plot =pylab)
pylab.show()

#Using Sklearn
from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(X_train,y_train)
X_train.columns
lm.coef_