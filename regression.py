import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline
df =pd.read_csv(r"F:\M.tech\2nd sem\data\fuelconsumption.csv")
#df=df.drop(['Model year','Make','Model','Vehicle class','Fuel type','Transmission'],axis=1)
cdf = df[['Engine size (L)','Cylinders','Combined (L/100 km)','CO2 emissions (g/km)']]
cdf=cdf.rename(columns={"Engine size (L)":"ENGINESIZE","Cylinders":"CYLINDERS","Combined (L/100 km)":"FUELCONSUMPTION_COMB","CO2 emissions (g/km)":"CO2EMISSIONS"})

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
plt.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
p = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(p - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((p - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , p) )