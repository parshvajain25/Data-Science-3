#%%

# Importing Libraries

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

series = pd.read_csv('datasetA6_HP.csv')

#%%

# Question1

print()
print("Question 1")
print()
print("(a)")
print()

series.plot()
plt.xlabel('Dates')
plt.ylabel('Power Consumption (MU)')
plt.title('Line plot for the data')
plt.show()

print()
print("(b)")
print()

def autocorr(df, lag):
    autocorr = sm.tsa.acf(df, nlags = lag)
    return autocorr

Pearson_corr = autocorr(series['HP'], 1)
print('Between the generated one day lag time sequence and the given time', 'sequence is :', Pearson_corr[1])

print()
print("(c)")
print()

plt.figure()
plt.xlabel("t_th")
plt.ylabel("(t-1)_th")
plt.scatter(x = series.iloc[1:500]["HP"], y = series.iloc[0:499]["HP"])
plt.show()

print()
print("(d)")
print()

Corr_list = []
for i in range(1,8):
    Pearson_corr = autocorr(series['HP'], i)
    Corr_list.append(Pearson_corr[i])
    print('Correlation Coefficient for lag={} is'.format(i), Pearson_corr[i])

plt.plot(Corr_list,color = 'red', linestyle = 'dashed', linewidth = 2, marker ='s', markerfacecolor = 'black', markersize = 9)
plt.xlabel('Lagged value')
plt.ylabel('Pearson Correlation')
plt.title('Line plot for the data')
plt.show()

print()
print("(e)")
print()

acf = pd.plotting.autocorrelation_plot(series.iloc[1:500]["HP"]) 
acf.plot()
plt.xlabel('Lagged value')
plt.ylabel('Pearson Correlation')
plt.show()

#%%

# Question2

print()
print("Question 2")
print()

train, test = series[0:-250], series[-250:]

test = pd.DataFrame(test)
test.columns = ["Date","HP"]
MSE = mse(test["HP"][1:250], test["HP"][0:249])
print("RMSE =",round(MSE**0.5,5))

#%%

# Question3

print()
print("Question 3")
print()
print("(a)")
print()

train, test = series[1:251],series[251:]
model = AutoReg(train["HP"], lags=5)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
predictions = model_fit.predict(start = len(train), end = len(train) + len(test), dynamic = False).iloc[:-1]
rmse = np.sqrt(mse(test.HP, predictions))
pred = list(predictions)
test1 = list(test["HP"])
plt.plot(test1,pred)
plt.xlabel('Original Data')
plt.ylabel('Predicted Data')
plt.show()
print("For lag = 5 Test RMSE: %.5f" % rmse)

print()
print("(b)")
print()

rmse_list = []
for l in [1,5,10,15,25]:   
    model = AutoReg(train["HP"], lags=l)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test), dynamic=False).iloc[:-1]
    rmse_list.append(np.sqrt(mse(test.HP, predictions)))
    print("For lag =",l,"Test RMSE: %.5f" % rmse_list[-1])
    
print()
print("(c)")
print()

optimal_l = 0
for l in range(1,25):
    autocorr =  train["HP"].autocorr(lag = l)
    if (np.absolute(autocorr)) - (2/np.sqrt(250-l)) <0 :
        optimal_l = l
        break
model = AutoReg(train["HP"], lags=l)
model_fit = model.fit()
predictions = model_fit.predict(start = len(train), end = len(train) + len(test), dynamic = False).iloc[:-1]
rmse = np.sqrt(mse(test.HP, predictions))
print("For lag =", l, "Test RMSE: %.5f" % rmse)
  
#%%      

