#%%

# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m

#%%

# Reading the csv files

df_miss = pd.read_csv("pima_indians_diabetes_miss.csv")             

df_original =pd.read_csv("pima_indians_diabetes_original.csv")             

#%%

## Question 1

print('Question 1: Plot a graph of the attribute names (x-axis) with the number of missing values in them (y-axis).')
print()

nan_val = df_miss.isnull().sum() # isnull() return True if empty, and False of filled; sum() is the count of Nan values)

plt.plot(nan_val)
plt.xlabel("Attributes")
plt.ylabel("Number of NaN values")
plt.show() 
 
#%%

## Question 2

print()
print('Question 2')
print()
print('(A) Delete (drop) the tuples (rows) having equal to or more than one third of attributes with missing values. Print the total number of tuples deleted and also print the row numbers of the deleted tuples (with respect to pima_indians_diabetes_miss.csv). ')
print()
 
df_miss_a=df_miss.dropna(thresh=7) # Total 9 attributes, so drop if 3 or more NaN values i.e. 7 is the threshold value

print("Number of rows deleted:", df_miss.shape[0] - df_miss_a.shape[0])

rows_del_a = []
for i in range(0, df_miss.shape[0]):
    if df_miss.isnull().sum(axis=1)[i]>=3:
        rows_del_a.append(i+2)    # shifting by 2 to match row number in .csv file 
        
print("Row numbers of the deleted tuples: ", rows_del_a)
       
print()
print('(B)  Drop the tuples (rows) having missing value in the target (class) attribute. Print the total number of tuples deleted and also print the row numbers of the deleted tuples (with respect to pima_indians_diabetes_miss.csv).')
print()   
 
rows_del_b = []
for i in df_miss[df_miss['class'].isnull()].index:
    rows_del_b.append(i+2)

print("Number of tuples deleted: ", len(rows_del_b))
print("Row numbers of the deleted tuples: ", rows_del_b)
 
#%%                
        
# Question 3

print()
print('Question 3: After step 2, count and print the number of missing values in each attributes. Also find and print the total number of missing values in the file (after the deletion of tuples).')
print()

df_miss_a = df_miss.dropna(thresh=7)
df_miss_b = df_miss_a.dropna(subset=["class"])

print("NaN values attribute wise:")
print(df_miss_b.isnull().sum())
print("Total NaN values in the file:")
print(df_miss_b.isnull().sum().sum())

#%%

# Question 4

print()
print('Question 4: Experiments on filling missing values:')
print()
print('(A) Replace the missing values by mean of their respective attribute.')
print()
print('(i) Compute the mean, median, mode and standard deviation for each attributes and compare the same with that of the original file.')
print()

df_filled_4a=df_miss.fillna({
        "pregs":df_miss_b["pregs"].mean(),
        "plas":df_miss_b["plas"].mean(),
        "pres":df_miss_b["pres"].mean(),
        "skin":df_miss_b["skin"].mean(),
        "test":df_miss_b["test"].mean(),
        "BMI":df_miss_b["BMI"].mean(),
        "pedi":df_miss_b["pedi"].mean(),
        "Age":df_miss_b["Age"].mean(),
        "class":df_miss_b["class"].mean(),
        })
 
# Mean of new dataframe and original   
print("New Mean:", df_filled_4a.mean())
print("Old Mean:", df_original.mean())

# Median of new dataframe and original
print("New Median:", df_filled_4a.median())
print("Old Median:", df_original.median())

# Mode of new dataframe and original
print("New Mode:", df_filled_4a.mode())
print("Old Mode:", df_original.mode())

# Standard Deviation of new dataframe and original
print("New Std:", df_filled_4a.std())
print("Old Std:", df_original.std())

print()
print('(ii) Calculate the root mean square error (RMSE) between the original and replaced values for each attribute. (Get original values from original file provided). Compute RMSE using the equation (1). Plot these RMSE with respect to the attributes.')
print()

# Function to find RMSE values for attributes
def RMSE_a(a):
    x = 0
    for i in range(0,767):
        x = x + ((df_original[a][i] - df_filled_4a[a][i])**2)
        i=i+1
    return m.sqrt(x/767)

RMSE_val_4a=[] 

#RMSE value of all attributes
print("RMSE value for pregs:",RMSE_a("pregs"))
RMSE_val_4a.append(RMSE_a("pregs"))    

print("RMSE value for plas:",RMSE_a("plas"))
RMSE_val_4a.append(RMSE_a("plas"))    

print("RMSE value for pres:",RMSE_a("pres"))
RMSE_val_4a.append(RMSE_a("pres"))    

print("RMSE value for skin:",RMSE_a("skin"))
RMSE_val_4a.append(RMSE_a("skin"))    

print("RMSE value for test:",RMSE_a("test"))
RMSE_val_4a.append(RMSE_a("test"))

print("RMSE value for BMI:",RMSE_a("BMI"))
RMSE_val_4a.append(RMSE_a("BMI"))    
    
print("RMSE value for pedi:",RMSE_a("pedi"))
RMSE_val_4a.append(RMSE_a("pedi"))    

print("RMSE value for Age:",RMSE_a("Age"))
RMSE_val_4a.append(RMSE_a("Age"))    

print("RMSE value for class:",RMSE_a("class"))
RMSE_val_4a.append(RMSE_a("class"))    

plt.plot(df_original.columns,RMSE_val_4a)
plt.xlabel("Attributes")
plt.ylabel("RMSE value")
plt.show()

print()
print('(B) Replace the missing values in each attribute using linear interpolation technique. Use df.interpolate() with suitable arguments.')
print()
print('(i) Compute the mean, median, mode and standard deviation for each attributes and compare the same with that of the original file.')
print()

df_filled_4b = df_miss.interpolate(method="linear")

# Mean of new dataframe and original   
print("New Mean:", df_filled_4b.mean())
print("Old Mean:", df_original.mean())

# Median of new dataframe and original
print("New Median:", df_filled_4b.median())
print("Old Median:", df_original.median())

# Mode of new dataframe and original
print("New Mode:", df_filled_4b.mode())
print("Old Mode:", df_original.mode())

# Standard Deviation of new dataframe and original
print("New Std:", df_filled_4b.std())
print("Old Std:", df_original.std())

print()
print('(ii) Calculate the root mean square error (RMSE) between the original and replaced values for each attribute. (Get original values from original file provided). Compute RMSE using the equation (1). Plot these RMSE with respect to the attributes.')
print()

# Function to find RMSE values for attributes
def RMSE_b(a):
    x = 0
    for i in range(0,767):
        x = x + ((df_original[a][i] - df_filled_4b[a][i])**2)
        i=i+1
    return m.sqrt(x/767)

RMSE_val_4b=[] 

#RMSE value of all attributes
print("RMSE value for pregs:",RMSE_b("pregs"))
RMSE_val_4b.append(RMSE_b("pregs"))    

print("RMSE value for plas:",RMSE_b("plas"))
RMSE_val_4b.append(RMSE_b("plas"))    

print("RMSE value for pres:",RMSE_b("pres"))
RMSE_val_4b.append(RMSE_b("pres"))    

print("RMSE value for skin:",RMSE_b("skin"))
RMSE_val_4b.append(RMSE_b("skin"))    

print("RMSE value for test:",RMSE_b("test"))
RMSE_val_4b.append(RMSE_b("test"))

print("RMSE value for BMI:",RMSE_b("BMI"))
RMSE_val_4b.append(RMSE_b("BMI"))    
    
print("RMSE value for pedi:",RMSE_b("pedi"))
RMSE_val_4b.append(RMSE_b("pedi"))    

print("RMSE value for Age:",RMSE_b("Age"))
RMSE_val_4b.append(RMSE_b("Age"))    

print("RMSE value for class:",RMSE_b("class"))
RMSE_val_4b.append(RMSE_b("class"))    

plt.plot(df_original.columns,RMSE_val_4b)
plt.xlabel("Attributes")
plt.ylabel("RMSE value")
plt.show()

#%%

# Question 5

print()
print('Question 5: Outlier Detection')
print()
print('(i) After replacing the missing values by interpolation method, find the outliers in the attributes “Age” and “BMI”. Outliers are the values that does not satisfy the condition (Q1 – (1.5 * IQR)) < X < (Q3 + (1.5 * IQR)), where X is the value of the attribute, IQR is the inter quartile range, Q1 and Q3 are the first and third quartiles. Obtain the boxplot for these attributes. ')
print()

df5 = df_filled_4b

Q1_age = np.percentile(df5.Age,25)    # Quartile 1

Q2_age = np.percentile(df5.Age,50)    # Quartile 2 = median

Q3_age = np.percentile(df5.Age,75)    # Quartile 3 

# For removing outliers remove points lying outside 1.5*(Q3-Q1)
outlier_age=[]
for i in range(0,767):
    if df5["Age"][i] < Q1_age - 1.5*(Q3_age - Q1_age) or df5["Age"][i] > Q3_age + 1.5*(Q3_age - Q1_age):
        outlier_age.append(df5["Age"][i])
        
print("Outliers in Age attribute are as follows:", outlier_age)        
        

Q1_bmi = np.percentile(df5.BMI,25)    # Quartile 1

Q2_bmi = np.percentile(df5.BMI,50)    # Quartile 2 = median

Q3_bmi = np.percentile(df5.BMI,75)    # Quartile 3 


outlier_bmi=[]
for i in range(0,767):
    if df5["BMI"][i] < Q1_bmi - 1.5*(Q3_bmi - Q1_bmi) or df5["BMI"][i] > Q3_bmi + 1.5*(Q3_bmi - Q1_bmi):
        outlier_bmi.append(df5["BMI"][i])
        
print("Outliers in BMI attribute are as follows:", outlier_bmi)          

# Plotting boxplots
plt.title('Boxplot for Age')
plt.boxplot(df5['Age'])
plt.show()

plt.title('Boxplot for BMI')
plt.boxplot(df5['BMI'])
plt.show()

print()
print('(ii) Replace these outliers by the median of the attribute. Plot the boxplot again and observe the difference with that of the boxplot in (5i).')
print()

for i in range(0,767):
    if df5["Age"][i] < Q1_age - 1.5*(Q3_age - Q1_age) or df5["Age"][i] > Q3_age + 1.5*(Q3_age - Q1_age):
        df5["Age"][i] = Q2_age
    if df5["BMI"][i] < Q1_bmi - 1.5*(Q3_bmi - Q1_bmi) or df5["BMI"][i] > Q3_bmi + 1.5*(Q3_bmi - Q1_bmi):
        df5["BMI"][i] = Q2_bmi
        
# Plotting boxplots
plt.title('Boxplot for Age')
plt.boxplot(df5['Age'])
plt.show()

plt.title('Boxplot for BMI')
plt.boxplot(df5['BMI'])
plt.show()

















