#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


heart_data = pd.read_csv('heart.csv')
heart_data.head()


# In[3]:


heart_data.tail()


# In[4]:


heart_data['Sex']=heart_data['Sex'].map({"M":1,"F":0})


# In[5]:


heart_data["ChestPainType"] = heart_data["ChestPainType"].map({"TA":1,"ATA":2,"NAP":3,"ASY":4})


# In[6]:


heart_data["RestingECG"] = heart_data["RestingECG"].map({"Normal":1,"ST":2,"LVH":3})


# In[7]:


heart_data["ExerciseAngina"] = heart_data["ExerciseAngina"].map({"N":0,"Y":1})


# In[8]:


heart_data["ST_Slope"] = heart_data["ST_Slope"].map({"Up":1,"Down":2,"Flat":3})


# In[9]:


heart_data["Oldpeak"].unique()


# In[10]:


heart_data


# In[11]:


heart_data.describe()


# In[12]:


heart_data['HeartDisease'].value_counts()


# In[13]:


columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']

for column in columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=heart_data, x=column)
    plt.title(column)
    plt.show()


# In[14]:


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Detect outliers for each numerical feature
numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
outliers = {}
for column in numerical_columns:
    outliers[column] = detect_outliers_iqr(heart_data, column).shape[0]

print("Outliers detected per column:", outliers)


# In[15]:


for column in numerical_columns:
    sns.distplot(heart_data[column])
    plt.show()


# In[16]:


for column in numerical_columns:
    Q1 = heart_data[column].quantile(0.25)
    Q3 = heart_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    heart_data[column] = heart_data[column].clip(lower_bound, upper_bound)


# In[17]:


heart_data.shape


# In[18]:


columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']

for column in columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=heart_data, x=column)
    plt.title(column)
    plt.show()


# In[19]:


X = heart_data.drop(columns='HeartDisease', axis=1)
Y = heart_data['HeartDisease']


# In[20]:


print(X)


# In[21]:


print(Y)


# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[23]:


print(X.shape, X_train.shape, X_test.shape)


# In[26]:


model_xg = XGBClassifier()
model_xg.fit(X_train, Y_train)


# In[27]:


model_xg.score(X_train, Y_train)


# In[28]:


model_xg.score(X_test, Y_test)


# In[29]:


y_pred = model_xg.predict(X_test)


# In[30]:


rf_acc = accuracy_score(Y_test, y_pred)
print(rf_acc)


# In[ ]:


Age = int(input("Enter: "))
Sex = int(input("Enter: "))
cp = int(input("Enter:"))
trestbps = int(input("Enter: "))
chol = int(input("Enter: "))
fbs = int(input("Enter:"))
restecg = int(input("Enter:"))
thalch = int(input("Enter:"))
exang = int(input("Enter:"))
oldpeak = int(input("Enter:"))
slope  = float(input("Enter:"))

input_data = (Age,Sex,cp,trestbps,chol,fbs,restecg,thalch,exang,oldpeak,slope)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = rf_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


# ##### 
