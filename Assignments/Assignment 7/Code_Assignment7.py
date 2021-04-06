# %% 

# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy import spatial as spatial

#%%

# Reading mnist files

mtrain = pd.read_csv("mnist-tsne-train.csv")
mtest = pd.read_csv("mnist-tsne-test.csv")

mtrain_cluster = mtrain.iloc[:,0:2]
mtest_cluster = mtest.iloc[:,0:2]

# Calculating purity score

def KMEANS(K):
    print("K =", K)
    kmeans = KMeans(n_clusters = K, random_state = 42)
    kmeans.fit(mtrain_cluster)
    kmeans_prediction_train = kmeans.predict(mtrain_cluster)
    
    print()
    print("(a)")
    print()
    
    mtrain_cluster['cluster'] = kmeans_prediction_train
    sns.set_style('whitegrid')
    sns.lmplot('dimention 1','dimension 2', data = mtrain_cluster, hue = 'cluster', height = 6, aspect = 1, fit_reg = False)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c ='black', s = 200, alpha = 1)
    plt.title("K-Means on train data, K = " + str(K))
    plt.show(sns)
    
    print()
    print("(b)")
    print()
    
    pscore = purity_score(mtrain['labels'], kmeans_prediction_train)
    print("Purity score with K-Means on train data:", pscore)
    
    print()
    print("(c)")
    print()
    
    kmeans_prediction_test = kmeans.predict(mtest_cluster)
    mtest_cluster['cluster'] = kmeans_prediction_test
    sns.set_style('whitegrid')
    sns.lmplot('dimention 1','dimension 2', data = mtest_cluster, hue = 'cluster', height = 6, aspect = 1, fit_reg = False)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c ='black', s = 200, alpha = 1)
    plt.title("K-Means on test data, K = " + str(K))
    plt.show(sns)
    
    print()
    print("(d)")
    print()
    
    pscore = purity_score(mtest['labels'], kmeans_prediction_test)
    print("Purity score with K-Means on test data:", pscore)
    
def GMM(K):
    print("K =", K)
    gmm = GaussianMixture(n_components = K, random_state = 42)
    gmm.fit(mtrain_cluster)
    GMM_prediction_train = gmm.predict(mtrain_cluster)
    
    print()
    print("(a)")
    print()
    
    mtrain_cluster['cluster'] = GMM_prediction_train
    sns.set_style('whitegrid')
    sns.lmplot('dimention 1','dimension 2', data = mtrain_cluster, hue = 'cluster', height = 6, aspect = 1, fit_reg = False)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c = 'black', s = 200, alpha = 1)
    plt.title("GMM on train data, K = " + str(K))
    plt.show(sns)
    
    print()
    print("(b)")
    print()
    
    pscore = purity_score(mtrain['labels'], GMM_prediction_train)
    print("Purity score with GMM on train data:", pscore)
    
    print()
    print("(c)")
    print()
    
    GMM_prediction_test = gmm.predict(mtest_cluster)
    mtest_cluster['cluster'] = GMM_prediction_test
    sns.set_style('whitegrid')
    sns.lmplot('dimention 1','dimension 2', data = mtest_cluster, hue = 'cluster', height = 6, aspect = 1, fit_reg = False)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c = 'black', s = 200, alpha = 1)
    plt.title("GMM on test data, K = " + str(K))
    plt.show(sns)
    
    print()
    print("(d)")
    print()
    
    pscore = purity_score(mtest['labels'], GMM_prediction_test)
    print("Purity score with GMM on test data:", pscore)

def dbscan_predict(dbscan_model, X_new, metric = spatial.distance.euclidean):
        y_new = np.ones(shape = len(X_new), dtype = int)*-1 
        for j, x_new in enumerate(X_new):
            for i, x_core in enumerate(dbscan_model.components_):
                if metric(x_new, x_core) < dbscan_model.eps:
                    y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                    break
        return y_new    

def DBSCANN(e, m):
    print("Eps =", e, "and Min_Samples =", m)
    dbscan_model = DBSCAN(eps = e, min_samples = m).fit(mtrain_cluster)
    DBSCAN_prediction_train = dbscan_model.labels_
    
    print()
    print("(a)")
    print()
    
    mtrain_cluster['cluster'] = DBSCAN_prediction_train
    sns.set_style('whitegrid')
    sns.lmplot('dimention 1','dimension 2', data = mtrain_cluster, hue = 'cluster', height = 6, aspect = 1, fit_reg = False)
    plt.title("Eps = " + str(e) + " and Min_Samples = " + str(m) + " Train")
    plt.show(sns)
    
    print()
    print("(b)")
    print()
    
    pscore = purity_score(mtrain['labels'], DBSCAN_prediction_train)
    print("Purity score with DBSCAN on train data:", pscore)
    
    print()
    print("(c)")
    print()
    
    DBSCAN_prediction_test = dbscan_predict(dbscan_model, mtest_cluster.values, metric = spatial.distance.euclidean)
    
    mtest_cluster['cluster'] = DBSCAN_prediction_test
    sns.set_style('whitegrid')
    sns.lmplot('dimention 1','dimension 2', data = mtest_cluster, hue = 'cluster', height = 6, aspect = 1, fit_reg = False)
    plt.title("Eps = " + str(e) + " and Min_Samples = " + str(m) + " Test")
    plt.show(sns)
    
    print()
    print("(d)")
    print()
    
    pscore = purity_score(mtest['labels'], DBSCAN_prediction_test)
    print("Purity score with DBSCAN on test data:", pscore)

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
        
#%%

# Question 1

print()
print("Question 1")
print()

mtrain_cluster = mtrain.iloc[:,0:2]
mtest_cluster = mtest.iloc[:,0:2]

KMEANS(10)

#%%

# Question 2

print()
print("Question 2")
print()

mtrain_cluster = mtrain.iloc[:,0:2]
mtest_cluster = mtest.iloc[:,0:2]

GMM(10)

#%%

# Question 3

print()
print("Question 3")
print()

mtrain_cluster = mtrain.iloc[:,0:2]
mtest_cluster = mtest.iloc[:,0:2]
    
DBSCANN(5, 10)

#%%

# BONUS QUESTIONS

print()
print("BONUS QUESTIONS")
print()

#%%

# Question A

print()
print("Question A")
print()
print("K-Means")
print()

K = [2,5,8,12,18,20]

for i in K:
    mtrain_cluster = mtrain.iloc[:,0:2]
    mtest_cluster = mtest.iloc[:,0:2]
    KMEANS(i)

K = [2,5,8,12,18,20]

distortions = [] 
mapping1 = {} 
  
for k in K: 
    kmeans = KMeans(n_clusters = k).fit(mtrain_cluster) 
    kmeans.fit(mtrain_cluster)     
    distortions.append(sum(np.min(cdist(mtest_cluster, kmeans.cluster_centers_, 'euclidean'),axis=1)) / mtrain_cluster.shape[0])  
    mapping1[k] = sum(np.min(cdist(mtest_cluster, kmeans.cluster_centers_,'euclidean'),axis=1)) / mtest_cluster.shape[0] 

for key,val in mapping1.items(): 
    print(str(key)+' : '+str(val))

plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion (K-Means)') 
plt.show()

print()
print("Optimal value of K using K-Means =", 8)
print()
print("GMM")
print()

K = [2,5,8,12,18,20]

for i in K:
    mtrain_cluster = mtrain.iloc[:,0:2]
    mtest_cluster = mtest.iloc[:,0:2]
    GMM(i)

distortions = []  
mapping1 = {} 
  
for k in K: 
    gmm = GaussianMixture(n_components = k).fit(mtrain_cluster) 
    gmm.fit(mtrain_cluster)     
    distortions.append(sum(np.min(cdist(mtest_cluster, gmm.means_, 'euclidean'),axis=1)) / mtrain_cluster.shape[0]) 
    mapping1[k] = sum(np.min(cdist(mtest_cluster, gmm.means_,'euclidean'),axis=1)) / mtest_cluster.shape[0] 
for key,val in mapping1.items(): 
    print(str(key)+' : '+str(val))

plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion(GMM)') 
plt.show()

print()
print("Optimal value of K using GMM =", 12)
print()

#%%

# Question B

print()
print("Question B")
print()

eps = [1, 5, 10]
min_samples = [10, 30, 50]

for e in eps:
    for m in min_samples:
        
        mtrain_cluster = mtrain.iloc[:,0:2]
        mtest_cluster = mtest.iloc[:,0:2]
  
        DBSCANN(e, m)
        
#%%











