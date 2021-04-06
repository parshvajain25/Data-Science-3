#%%

# Importing the required modules

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the csv file
df = pd.read_csv('seismic_bumps1.csv')

X=pd.concat([df[df.columns[:8]], df[df.columns[16:18]]],axis=1)
Y=df[df.columns[-1]]
df = pd.concat([X,Y], axis=1)
df_0 = df[df["class"] == 0]
[X_train_0, X_test_0, X_label_train_0, X_label_test_0] = train_test_split(df_0[df_0.columns[:-1]], df_0[df_0.columns[-1]], test_size=0.3, random_state=42,shuffle=True)

df_1 = df[df["class"]==1]
[X_train_1, X_test_1, X_label_train_1, X_label_test_1] = train_test_split(df_1[df_1.columns[:-1]], df_1[df_1.columns[-1]], test_size=0.3, random_state=42,shuffle=True)

#%%

# Question1

print("Question 1")
print()

test_0 = pd.concat([X_test_0, X_label_test_0], axis=1)
test_1 = pd.concat([X_test_1, X_label_test_1], axis=1)
test_A = pd.concat([test_0, test_1])

train_0 = pd.concat([X_train_0, X_label_train_0], axis=1)
train_1 = pd.concat([X_train_1, X_label_train_1], axis=1)
train_A = pd.concat([train_0, train_1])

test_A.to_csv('seismic-bumps-test.csv', index=False)
train_A.to_csv('seismic-bumps-train.csv', index=False)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_A[train_A.columns[:-1]], train_A[train_A.columns[-1]])
Y_pred_1 = neigh.predict(test_A[test_A.columns[:-1]])

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_A[train_A.columns[:-1]], train_A[train_A.columns[-1]])
Y_pred_3 = neigh.predict(test_A[test_A.columns[:-1]])

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_A[train_A.columns[:-1]], train_A[train_A.columns[-1]])
Y_pred_5 = neigh.predict(test_A[test_A.columns[:-1]])

print('(A)')
print()

from sklearn.metrics import confusion_matrix

print("Confusion Matrix for k=1:")
print(confusion_matrix(test_A[test_A.columns[-1]], Y_pred_1))

print("Confusion Matrix for k=3:")
print(confusion_matrix(test_A[test_A.columns[-1]], Y_pred_3))

print("Confusion Matrix for k=5:")
print(confusion_matrix(test_A[test_A.columns[-1]], Y_pred_5))

print('(B)')
print()

from sklearn.metrics import accuracy_score

c_acc_1 = accuracy_score(test_A[test_A.columns[-1]], Y_pred_1)
c_acc_3 = accuracy_score(test_A[test_A.columns[-1]], Y_pred_3)
c_acc_5 = accuracy_score(test_A[test_A.columns[-1]], Y_pred_5)

print("Classification Accuracy for k=1: ", c_acc_1)
print("Classification Accuracy for k=3: ", c_acc_3)
print("Classification Accuracy for k=5: ", c_acc_5)

#%%

# Question 2

print()
print("Question 2")
print()

def min_max(df, df1):  # Function for normalisation
    df_norm = df.copy()

    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df1[column].min()) / (df1[column].max() - df1[column].min())
        df_norm[column] = df_norm[column] * (1)
    return df_norm


X_train_normalise = min_max(train_A[train_A.columns[:-1]], train_A[train_A.columns[:-1]])
X_test_normalise = min_max(test_A[test_A.columns[:-1]], train_A[train_A.columns[:-1]])

test_normalise_B = pd.concat([X_test_normalise, test_A[test_A.columns[-1]]], axis=1)
train_normalise_B = pd.concat([X_train_normalise, train_A[train_A.columns[-1]]], axis=1)
test_normalise_B.to_csv('seismic-bumps-test-normalised.csv', index=False)
train_normalise_B.to_csv('seismic-bumps-train-Normalised.csv', index=False)

max_acc = 0
k_max_acc = 1
for k in range(1,6,2):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train_normalise, train_A[train_A.columns[-1]])
    Y_pred = neigh.predict(X_test_normalise)
    print("Confusion Matrix for k=",k)
    print(confusion_matrix(test_A[test_A.columns[-1]], Y_pred))
    c_acc = accuracy_score(test_A[test_A.columns[-1]], Y_pred)
    if c_acc>max_acc:
        k_max_acc = k
        max_acc = c_acc
    print("Classification Accuracy for k= ",k,"is :", c_acc)
print()
print("Maximum accuracy is with K= ",k_max_acc) 

#%%

# Question 3

print()
print("Question 3")
print()

from sklearn.mixture import GaussianMixture

X_train_class_0 = train_0.drop(columns=["class"])
X_train_class_1 = train_1.drop(columns=["class"])

mean_0 = pd.DataFrame(X_train_class_0.mean())
mean_0 = mean_0.T
mean_1 = pd.DataFrame(X_train_class_1.mean())
mean_1 = mean_1.T
print("Mean for class 0: ", mean_0)
print("Mean for class 1: ", mean_1)
print()

cov_0 = np.cov(X_train_class_0.values.T)
cov_1 = np.cov(X_train_class_1.values.T)
print(cov_0)
print(cov_1)

gmm = GaussianMixture(n_components=1)

gmm.fit(X_train_class_0)
cov_0_gmm = gmm.covariances_

gmm.fit(X_train_class_1)
cov_1_gmm = gmm.covariances_

def mahalanobis(X, mean, cov):
    X = X - mean
    l = np.dot(X, np.linalg.inv(cov))
    return np.dot(l, X.T)

def likelihood(X, mean, cov):
    exponent = np.exp(-(0.5) * mahalanobis(X, mean, cov))
    return (1 / (np.sqrt(2 * np.pi) * np.sqrt(np.linalg.det(cov)))) * exponent

prior_0 = len(X_train_class_0) / (len(X_train_class_1) + len(X_train_class_0))
prior_1 = len(X_train_class_1) / (len(X_train_class_1) + len(X_train_class_0))

def P(X):
    return prior_0 * likelihood(X, mean_0, cov_0_gmm) + prior_1 * likelihood(X, mean_1, cov_1_gmm)

def posterior_probability(X, prior, mean, cov):
    return (prior * likelihood(X, mean, cov)) / P(X)

y_pred_bayes = []
for i in range(len(test_A[test_A.columns[-1]])):
    Y_0 = posterior_probability(test_A[test_A.columns[:-1]].iloc[i], prior_0, mean_0, cov_0_gmm)
    Y_1 = posterior_probability(test_A[test_A.columns[:-1]].iloc[i], prior_1, mean_1, cov_1_gmm)
    if (Y_0 > Y_1):
        y_pred_bayes.append(0)
    if (Y_1 > Y_0):
        y_pred_bayes.append(1)
        
print(y_pred_bayes.count(0), y_pred_bayes.count(1))
Y_pred_bayes = pd.DataFrame(y_pred_bayes, columns=["class"])

print("Confusion Matrix for Bayes Classifier:")
print(confusion_matrix(test_A[test_A.columns[-1]], Y_pred_bayes))

print("Classification Accuracy for Bayes Classifier:")
print(accuracy_score(test_A[test_A.columns[-1]], Y_pred_bayes))

c_bayes = accuracy_score(test_A[test_A.columns[-1]], Y_pred_bayes)


