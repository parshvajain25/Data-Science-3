# %% 

# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

# Reading landslide_data3.csv

landslide_data = pd.read_csv('landslide_data3.csv')

# %% 

## Question 1

print('Question 1: Mean, median, mode, minimum, maximum and standard deviation for all the attributes.  ')
print()

mean_ls = landslide_data.mean()       # Mean of every attribute of the dataset
median_ls = landslide_data.median()   # Median of every attribute of the dataset
mode_ls = landslide_data.mode()       # Mode of every attribute of the dataset
std_ls = landslide_data.std()         # Standar Deviation of every attribute of the dataset
max_ls = landslide_data.max()         # Maximum of every attribute of the dataset
min_ls = landslide_data.min()         # Minimum of every attribute of the dataset

attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture' ] # In order given in the csv file

Q1_data = {}
for i in attributes:
    Q1_data[i] = [  mean_ls[i],
                    median_ls[i],
                    mode_ls[i][0],
                    min_ls[i],
                    max_ls[i],
                    std_ls[i]
                 ]       

Q1_ans = pd.DataFrame(Q1_data, index=['Mean', 'Median', 'Mode', 'Minimum', 'Maximum', 'Standard Deviation'])

print(Q1_ans)

#%% 

## Question 2

print()
print('Question 2: Obtain the scatter plot between')
print()
print('(A) ‘rain’ and each of the other attributes, excluding ‘dates’ and ‘stationid’. Consider ‘rain’ in x-axis and other attributes in y-axis.  ')
print()

X = landslide_data['rain']

print('Scatter plots with rain')
print()

# Rain vs Temperature Scatter Plot
Y = landslide_data['temperature']
plt.scatter(X, Y, color='orange')
plt.xlabel('Rain')
plt.ylabel('Temperature')
plt.title('Rain vs Temperature Scatter Plot')
plt.show()

# Rain vs Humidity Scatter Plot
Y = landslide_data['humidity']
plt.scatter(X, Y, color='blue')
plt.xlabel('Rain')
plt.ylabel('Humidity')
plt.title('Rain vs Humidity Scatter Plot')
plt.show()

# Rain vs Pressure Scatter Plot
Y = landslide_data['pressure']
plt.scatter(X, Y, color='green')
plt.xlabel('Rain')
plt.ylabel('Pressure')
plt.title('Rain vs Pressure Scatter Plot')
plt.show()

# Rain vs Lightavgw/o0 Scatter Plot
Y = landslide_data['lightavgw/o0']
plt.scatter(X, Y, color='magenta')
plt.xlabel('Rain')
plt.ylabel('Lightavgw/o0')
plt.title('Rain vs Lightavgw/o0 Scatter Plot')
plt.show()

# Rain vs Lightmax Scatter Plot
Y = landslide_data['lightmax']
plt.scatter(X, Y, color='red')
plt.xlabel('Rain')
plt.ylabel('Lightmax')
plt.title('Rain vs Lightmax Scatter Plot')
plt.show()

# Rain vs Moisture Scatter Plot
Y = landslide_data['moisture']
plt.scatter(X, Y, color='cyan')
plt.xlabel('Rain')
plt.ylabel('Moisture')
plt.title('Rain vs Moisture Scatter Plot')
plt.show()

print()
print('(B) ‘temperature’ and each of the other attributes, excluding ‘dates’ and ‘stationid’, Consider ‘temperature’ in x-axis and other attributes in y-axis.  ')
print()

X = landslide_data['temperature']

print('Scatter plots with Temperature')
print()

# Temperature vs Rain Scatter Plot
Y = landslide_data['rain']
plt.scatter(X, Y, color='orange')
plt.xlabel('Temperature')
plt.ylabel('Rain')
plt.title('Temperature vs Rain Scatter Plot')
plt.show()

# Temperature vs Humidity Scatter Plot
Y = landslide_data['humidity']
plt.scatter(X, Y, color='blue')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Temperature vs Humidity Scatter Plot')
plt.show()

# Temperature vs Pressure Scatter Plot
Y = landslide_data['pressure']
plt.scatter(X, Y, color='green')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.title('Temperature vs Pressure Scatter Plot')
plt.show()

# Temperature vs Lightavgw/o0 Scatter Plot
Y = landslide_data['lightavgw/o0']
plt.scatter(X, Y, color='magenta')
plt.xlabel('Temperature')
plt.ylabel('Lightavgw/o0')
plt.title('Temperature vs Lightavgw/o0 Scatter Plot')
plt.show()

# Temperature vs Lightmax Scatter Plot
Y = landslide_data['lightmax']
plt.scatter(X, Y, color='red')
plt.xlabel('Temperature')
plt.ylabel('Lightmax')
plt.title('Temperature vs Lightmax Scatter Plot')
plt.show()

# Temperature vs Moisture Scatter Plot
Y = landslide_data['moisture']
plt.scatter(X, Y, color='cyan')
plt.xlabel('Temperature')
plt.ylabel('Mositure')
plt.title('Temperature vs Moisture Scatter Plot')
plt.show()

#%% 

# Question 3

print()
print('Question 3: Find the value of correlation coefficient in the following cases:')
print()
print('(A) ‘rain’ with all other attributes (excluding dates and stationid)')
print()

attributes = ['temperature', 'humidity', 'pressure', 'lightavgw/o0', 'lightmax', 'moisture' ] 

print('Correlation coefficient of property with Rain: ')
for i in attributes:
    print(i, end="\t\t")
    cor_cof_rain = np.corrcoef(landslide_data['rain'], landslide_data[i])
    print(cor_cof_rain[0][1])

print()
print('(B) ‘temperature’ with all other attributes (excluding dates and stationid).')
print()

attributes = ['humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture' ]

print('Correlation coefficient of property with Temperature: ')
for i in attributes:
    print(i, end="\t\t")
    cor_cof_temp = np.corrcoef(landslide_data['temperature'], landslide_data[i])
    print(cor_cof_temp[0][1])

#%% 

# Question 4

print()
print('Question 4: Plot the histogram for the attributes ‘rain’ and ‘moisture’ ')
print()

# Histogram for Rain distribution
print('Histogram for Rain distribution')
print()

landslide_data['rain'].hist()
plt.title('Histogram for Rain distribution')
plt.show()

# Histogram for Moisture distribution
print('Histogram for Moisture distribution')
print()

landslide_data['moisture'].hist()
plt.title('Histogram for Moisture distribution')
plt.show()

#%% 

# Question 5

print()
print('Question 5: Plot the histogram of attribute ‘rain’ for each of the 10 stations')
print()

landslide_data['rain'].hist(by=landslide_data['stationid'])
plt.show()

#%% 

# Question 6

print()
print('Question 6: Obtain the boxplot for the attributes ‘rain’ and ‘moisture’')
print()

# Boxplot for rain distribution
print('Boxplot for Rain distribution')
print()

plt.title('Boxplot for Rain distribution')
plt.boxplot(landslide_data['rain'])
plt.show()

# Boxplot for moisture distribution
print('Boxplot for Moisture distribution')
print()

plt.title('BoxPlot for moisture distribution')
plt.boxplot(landslide_data['moisture'])
plt.show()