# Importing Libraries

import sklearn
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% 

# Reading landslide_data3.csv

df = pd.read_csv('landslide_data3.csv')
print(df)

#%%

# Cleaning and Removing the Outliers
# DO THIS ONLY ONE TIME
# Replacing outliers with median

q1 = df.quantile(0.25)
q2 = df.quantile(0.5)
q3 = df.quantile(0.75)

num_df = pd.DataFrame()

for col in df:
    s = df[col]
    if s.dtype == 'float64' or s.dtype == 'int64':
        # These are the only numeric rows in the data
        # Now Removing the outliers
        iqr = q3[col] - q1[col]
        s = s.where(
            (s > (q1[col] - 1.5*iqr)), q2[col])
        s = s.where(
            (s < (q3[col] + 1.5*iqr)), q2[col])
        df[col] = s
        num_df[col] = s

print(df)
print()
print(num_df)

#%% 

# Question 1

print('Question 1: Consider only last seven attributes (i.e. excluding dates and stationed) for the analysis. Replace the outliers (if at all any) in any attribute with the median of the respective attributes and do the following on outlier corrected data: ')
print()
print('A: Do the Min-Max normalization of the data (landslide_data3.csv) to scale the attribute values in the range 3 to 9. Find the minimum and maximum values before and after performing the Min-Max normalization of the attributes.')
print()

min_dict = num_df.min()
max_dict = num_df.max()

print("Minimum values before Normalising are: ")
print(min_dict, '\n')
print("Maximum values before Normaising are: ")
print(max_dict, '\n')

# Normalising the data b/w 3 and 9
diff = max_dict - min_dict
df1a = ((num_df - min_dict) / diff)*6 + 3

print("Minimum values after Normalising are: ")
print(df1a.min(), '\n')
print("Maximum values after Normalising are: ")
print(df1a.max(), '\n')

print()
print('B: Find the mean and standard deviation of the attributes of the data (landslide_data3.csv). Standardize each selected attribute using the relation 𝑥𝑥�n= (xn−μ)/σ where μ is mean and σ is standard deviation of that attribute. Compare the mean and standard deviations before and after the standardization.')
print()

df_mean = num_df.mean()
df_std = num_df.std()

print("Mean Values before Normalising are: ")
print(df_mean, '\n')
print("Standard Deviation values before Normaising are: ")
print(df_std, '\n')

# Standardising the data
df1b = (num_df - df_mean) / df_std

print("Mean values After Normalising are: ")
print(df1b.mean(), '\n')
print("Standard Deviation values After Normaising are: ")
print(df1b.std(), '\n')

#%% 

# Question 2

print()
print('Question 2: Generate 2-dimensional synthetic data of 1000 samples and let it be denoted as data matrix D of size 2x1000. Each sample is independently and identically distributed with bi-variate Gaussian distribution with user entered mean values, μ = [0, 0]T and covariance matrix, Σ = � 5 10 10 13�. Perform the followings: ')
print()
print('A: Draw a scatter plot of the data samples. ')
print()

cov_matrix = [[6.84806467, 7.63444163], [7.63444163, 13.02074623]]

df2 = np.random.multivariate_normal([0, 0], cov_matrix, 1000)
df2a = pd.DataFrame(df2)

plt.scatter(df2a[0], df2a[1], marker = 'o',s = 10, c = 'b')
plt.show()

print()
print('B: Compute the eigenvalues and eigenvectors of the covariance matrix and Plot the Eigen directions (with arrows/lines) onto the scatter plot of data ')
print()

eigen_values, eigen_vectors = np.linalg.eig(np.array(cov_matrix))
plt.scatter(df2a[0], df2a[1], marker = 'o',s = 10, c = 'b')
plt.quiver(0, 0, eigen_vectors[0][0], eigen_vectors[0][1], scale = 2, color = 'r')
plt.quiver(0, 0, eigen_vectors[1][0], eigen_vectors[1][1], scale = 5, color = 'r')
plt.show()

print()
print('C: Project the data on to the first and second Eigen direction individually and draw both the scatter plots superimposed on Eigen vectors')
print()

pca = PCA(n_components = 2)
pca_results = pca.fit_transform(df2a)
pca_dataframe = pd.DataFrame(pca_results)
plt.scatter(df2a[0], df2a[1], marker = 'o',s = 10, c = 'b')
plt.scatter(pca_dataframe[0], df2a[1], s=4, color = 'pink')
plt.quiver(0, 0, eigen_vectors[0][0], eigen_vectors[0][1], scale = 2, color = 'r')
plt.quiver(0, 0, eigen_vectors[1][0], eigen_vectors[1][1], scale = 5, color = 'r')
plt.show()

plt.scatter(df2a[0], df2a[1], marker = 'o',s = 10, c = 'b')
plt.scatter(pca_dataframe[1], df2a[1], s=6, marker = 'x', color = 'pink')
plt.quiver(0, 0, eigen_vectors[0][0], eigen_vectors[0][1], scale = 2, color = 'r')
plt.quiver(0, 0, eigen_vectors[1][0], eigen_vectors[1][1], scale = 5, color = 'r')
plt.show()

print()
print('D: Reconstruct the data samples using both eigenvectors, say it 𝐃𝐃�. Estimate the reconstruction error between 𝐃𝐃� and D using mean square error. ')
print()

pca_mse = pca.inverse_transform(pca_results)
mse_dataframe = pd.DataFrame(pca_mse)

mse = sklearn.metrics.mean_squared_error(df2a, mse_dataframe)
print('MSE: ', mse)


#%%

# Question 3

print()
print('Question 3: Perform principle component analysis (PCA) on outlier corrected standardized data (Data frame obtained after Qn 1b) and do the followings: ')
print()
print('A: Reduce the multidimensional (d = 7) data into lower dimensions (l = 2). Print the variance of the projected data along the two directions and compare with the eigenvalues of the two directions of projection. Also show the scatter plot of reduced dimensional data.')
print()

# Current DataFrame to work on df1b
df1b_corr = df1b.corr()
eigen_values, eigen_vectors = np.linalg.eig(df1b_corr)

eigen_list = list(eigen_values)
max_indices = []
for i in range(2):
    max_val = eigen_list[0]
    maxi = 0
    for j in range(len(eigen_list)):
        if eigen_list[j] > max_val:
            max_val = eigen_list[j]
            maxi = j
    print("Max eigen value is: ", max_val)
    max_indices.append(maxi + i)
    eigen_list.pop(maxi)

# Now using scikit PCA
pca = PCA(n_components=2)
pca.fit(df1b)

print("Variance ratio in eigen vectors: ")
print(pca.explained_variance_ratio_)

reduced_data = pca.fit_transform(df1b)
df3a = pd.DataFrame(reduced_data, columns=['A', 'B'])
plt.scatter(df3a['A'], df3a['B'])
plt.xlabel('A')
plt.ylabel('B')
plt.title('PCA reduced Scatter Plot')
plt.show()

print("Variance found of the reduced Data is: ")
print(df3a.var())

print()
print('B: Plot all the eigenvalues in the descending order ')
print()

eigen_list = list(eigen_values)
eigen_list.sort(reverse=True)
x = [i for i in range(len(eigen_list))]
plt.bar(x, eigen_list)
plt.xlabel('Index')
plt.ylabel('Eigen value')
plt.title('Eigen Values in Decreasing order Bar Plot')
plt.show()

print()
print('C: Plot the reconstruction errors in terms of RMSE considering the different values of l (=1, 2, ..., 7). The x-axis is the l and y-axis is reconstruction error in RMSE.')
print()

RMSE = []
for i in range(1, 8):
    pca = PCA(n_components=i)
    pca_results = pca.fit_transform(df1b)
    pca_proj_back = pca.inverse_transform(pca_results)
    # Usign the Numpy norm method to calculate the error
    total_loss = np.linalg.norm((df1b-pca_proj_back), None)
    RMSE.append(total_loss)

x = [1,2,3,4,5,6,7]
y = RMSE
plt.plot(x,y)
plt.xlabel('l')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs Components')
plt.show()