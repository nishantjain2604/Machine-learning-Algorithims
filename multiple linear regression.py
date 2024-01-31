import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline
df =pd.read_csv(r"F:\M.tech\2nd sem\data\fuelconsumption.csv")
#df=df.drop(['Model year','Make','Model','Vehicle class','Fuel type','Transmission'],axis=1)
cdf = df[['Engine size (L)','Cylinders','City','Highway','CO2 emissions (g/km)']]
cdf=cdf.rename(columns={"Engine size (L)":"ENGINESIZE","Cylinders":"CYLINDERS","CO2 emissions (g/km)":"CO2EMISSIONS"})

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','City','Highway']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
from sklearn.metrics import r2_score
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','City','Highway']])
test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','City','Highway']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared Error (MSE) : %.2f"
      % np.mean((y_hat - test_y) ** 2))
print('Variance score: %.2f' % regr.score(test_x, test_y))