#%%

# Importing the required modules

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sbtrain = pd.read_csv("seismic-bumps-train.csv")
sbtest = pd.read_csv("seismic-bumps-test.csv")

#%%

# PART-A

print("PART-A")

# Question1

print()
print("Question 1")
print()

X_train_0 = sbtrain[sbtrain["class"]==0].drop(["class"],axis=1)
X_train_1 = sbtrain[sbtrain["class"]==1].drop(["class"],axis=1)
X_train_label = sbtrain["class"]
X_test = sbtest[sbtest.columns[:-1]]
X_test_label = sbtest["class"]

prior_0 = len(X_train_0) / (len(X_train_1) + len(X_train_0))
prior_1 = len(X_train_1) / (len(X_train_1) + len(X_train_0))

def likelihood(X,gmm):
    return np.exp(gmm.score_samples((X)))

def P(x0,x1):
    return prior_0*x0 + prior_1*x1

def posterior(X,gmm0,gmm1):
    y = []
    for i,j in zip(likelihood(X,gmm0),likelihood(X,gmm1)):
        a = i*prior_0/P(i,j)
        b = j*prior_1/P(i,j)
        if a>b:
            y.append(0)
        else:
            y.append(1)
    return y

import warnings

warnings. filterwarnings('ignore')

acc = 0
index = 0
q=[2,4,8,16]

for i in q:
    gmm0 = GaussianMixture(n_components = i, covariance_type='full', random_state = 42)
    gmm1 = GaussianMixture(n_components = i, covariance_type='full', random_state = 42)
    gmm0.fit(X_train_0)
    gmm1.fit(X_train_1)
    y_pred = posterior(X_test,gmm0,gmm1)
    print("Confusion Matrix for Q =",i)
    print(confusion_matrix(X_test_label, y_pred))
    print("Classification Accuracy for Q =", i)
    print(accuracy_score(X_test_label, y_pred))
    print()
    if accuracy_score(X_test_label, y_pred) > acc:
        acc = accuracy_score(X_test_label, y_pred)
        index = i

# Question2

print()
print("Question 2")
print()

l1 = ["k = 5","k_normalised = 5", "Bayes Classifier UniModal", "Bayes Classifier GMM (Multimodal) (Q = " + str(index) + ")" ]
l2 = [0.9317010309278351, 0.9291237113402062, 0.875, acc]
df = pd.DataFrame(l2, index = l1, columns = ["Classification Accuracy"])

print(df)

#%%

# PART-B

print("PART-B")

df_atm = pd.read_csv('atmosphere_data.csv')
[X_train, X_test, Y_train, Y_test] = train_test_split(df_atm.iloc[:, :-1], df_atm['temperature'], test_size=0.3, random_state=42, shuffle=True)

#%%

# Question1

print()
print("Question 1")
print()

print()
print("(a)")
print()

regressor = LinearRegression()
regressor.fit(X_train.iloc[:, 1].values.reshape(-1, 1), Y_train)
predictions = regressor.predict(X_test.iloc[:, 1].values.reshape(-1, 1))
plt.scatter(X_train.iloc[:, 1], Y_train, color="blue")
plt.plot(X_train.iloc[:, 1], regressor.predict(X_train.iloc[:, 1].values.reshape(-1, 1)), color='red')
plt.title("Best Fit Line on Training Data")
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.show()

print()
print("(b)")
print()

print("Prediction Accuracy on Training Data using RMSE: ", (mean_squared_error(Y_train, regressor.predict(X_train.iloc[:, 1].values.reshape(-1, 1)))**0.5))

print()
print("(c)")
print()

print("Prediction Accuracy on Test Data using RMSE: ", (mean_squared_error(predictions, Y_test))**0.5)

print()
print("(d)")
print()

plt.scatter(Y_test, predictions, color="blue")
plt.title("Actual Temp. v/s Predicted Temp.")
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.show()

#%%

# Question2

print()
print("Question 2")
print()

def polynomial_predict(p, x_train, y_train, test_data):
    polynomial_features = PolynomialFeatures(p)
    x_poly = polynomial_features.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    test_data = polynomial_features.fit_transform(test_data)
    predict = regressor.predict(test_data)
    return predict

def create_rsme_plots(x_train, y_train, x_test, y_test, color='blue', plot_title='RSME Error on Test DATA'):
    rmse_list = []
    min_Rmse = 100
    best_fit_p = 2
    for p in range(2, 6):
        pred = polynomial_predict(p, x_train, y_train, x_test)
        mse = mean_squared_error(pred, y_test)
        rmse_list.append(mse**0.5)
        if min_Rmse > rmse_list[-1]:
            min_Rmse = rmse_list[-1]
            best_fit_p = p
    plt.bar(range(2, 6), rmse_list, color=color, align='center', width=[0.35, 0.35, 0.35, 0.35])
    plt.xticks(range(2, 6), range(2, 6))
    plt.xlabel('Degree of Polynomial (p)')
    plt.ylabel('RMSE')
    plt.title(plot_title)
    plt.show()
    return best_fit_p

print()
print("(a)")
print()

pressures_train = X_train.iloc[:, 1].values.reshape(-1, 1)
best_train_p = create_rsme_plots(pressures_train, Y_train, pressures_train, Y_train,'red', 'RSME Error on Training DATA')

print()
print("(b)")
print()

pressures_test = X_test.iloc[:, 1].values.reshape(-1, 1)
best_test_p = create_rsme_plots(pressures_train, Y_train, pressures_test, Y_test,'purple', 'RSME Error on Test DATA')

print()
print("(c)")
print()

test_pred = polynomial_predict(best_test_p, pressures_train, Y_train, pressures_test)

print(" Value of P with best test fit is: ", best_test_p)
pressures_test_1d = X_test.iloc[:, 1]
pol = np.polyfit(pressures_test_1d, test_pred, best_test_p)
x_values = np.linspace(pressures_train.min(), pressures_train.max(), 400).reshape(-1, 1)
Y = np.polyval(pol, x_values)
plt.figure()
plt.plot(x_values, Y, color='red')
plt.scatter(X_train.iloc[:, 1], Y_train, color="blue")
plt.title("Best Fit Curve on Training Data")
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.show()

print()
print("(d)")
print()

plt.scatter(Y_test, test_pred, color="blue")
plt.title("Actual Temp. v/s Predicted Temp.")
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.axis('equal')
plt.show()