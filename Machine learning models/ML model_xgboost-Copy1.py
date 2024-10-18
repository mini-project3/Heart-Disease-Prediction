#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# to visualize the dataset
import matplotlib.pyplot as plt
import seaborn as sns

# To preprocess the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
# import iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
#for classification tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import KBinsDiscretizer
#metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score


# In[2]:


# load the data from csv file placed locally in our pc
df = pd.read_csv('heart_disease_uci.csv')

# print the first 5 rows of the dataframe
df.head()


# In[3]:


df.drop(['id','dataset','ca','thal'], axis=1, inplace=True)
df.info()


# In[4]:


df


# In[5]:


sns.histplot(df['age'], kde=True)


# In[6]:


df.groupby('sex')['age'].value_counts()


# In[7]:


df['sex']=df['sex'].map({"Female":0,"Male":1})


# In[8]:


df["cp"].unique()


# In[9]:


df["cp"] = df["cp"].map({"typical angina":1,"atypical angina":2,"non-anginal":3,"asymptomatic":4})


# In[10]:


df["restecg"].unique()


# In[11]:


df["restecg"] = df["restecg"].map({"normal":1,"st-t abnormality":2,"lv hypertrophy":3})


# In[12]:


df["exang"] = df["exang"].replace({True: 1, False: 0})


# In[13]:


df["slope"].unique()


# In[14]:


df["slope"] = df["slope"].map({"upsloping":0,"downsloping":1,"flat":2})


# In[15]:


df["fbs"] = df["fbs"].replace({True: 1, False: 0})


# In[16]:


numeric_cols = ['oldpeak', 'thalch', 'chol', 'trestbps', 'age','slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']


# In[17]:


df


# In[18]:


df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
missing_data_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
missing_data_cols


# In[19]:


def impute_categorical_missing_data(passed_col):
    
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    
    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    if passed_col in bool_cols:
        y = label_encoder.fit_transform(y)
        
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    print("The feature '"+ passed_col+ "' has been imputed with", round((acc_score * 100), 2), "accuracy\n")

    X = df_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
                
    if len(df_null) > 0: 
        df_null[passed_col] = rf_classifier.predict(X)
        if passed_col in bool_cols:
            df_null[passed_col] = df_null[passed_col].map({0: False, 1: True})
        else:
            pass
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])
    
    return df_combined[passed_col]

def impute_continuous_missing_data(passed_col):
    
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    
    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
    
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor()

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    print("MAE =", mean_absolute_error(y_test, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_test, y_pred), "\n")

    X = df_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
                
    if len(df_null) > 0: 
        df_null[passed_col] = rf_regressor.predict(X)
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])
    
    return df_combined[passed_col]


# In[20]:


df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)


# In[21]:


for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((df[col].isnull().sum() / len(df)) * 100, 2))+"%")
    
    if col in numeric_cols:
        df[col] = impute_continuous_missing_data(col)
    else:
        pass


# In[22]:


df.isnull().sum()


# In[23]:


df.shape


# In[24]:


df


# In[25]:


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Detect outliers for each numerical feature
numerical_columns = ['age', 'restecg', 'chol', 'thalch', 'oldpeak']
outliers = {}
for column in numerical_columns:
    outliers[column] = detect_outliers_iqr(df, column).shape[0]

print("Outliers detected per column:", outliers)


# In[26]:


columns = ['age', 'restecg', 'chol', 'thalch', 'oldpeak']

for column in columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=column)
    plt.title(column)
    plt.show()


# In[27]:


for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)


# In[28]:


columns =['age', 'restecg', 'chol', 'thalch', 'oldpeak']


for column in columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=column)
    plt.title(column)
    plt.show()


# In[ ]:





# 

# In[29]:


X = df.drop(columns='num', axis=1)
Y = df['num']


# In[30]:


print(X)


# In[31]:


print(Y)


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[33]:


print(X.shape, X_train.shape, X_test.shape)


# In[34]:


model_xg = XGBClassifier()
model_xg.fit(X_train, Y_train)


# In[35]:


model_xg.fit(X_train, Y_train)


# In[36]:


model_xg.score(X_train, Y_train)


# In[37]:


model_xg.score(X_test, Y_test)


# In[38]:


y_pred = model_xg.predict(X_test)


# In[39]:


acc = accuracy_score(Y_test, y_pred)
print(acc)


# In[40]:


Age = int(input("Enter"))
Sex = int(input("Enter"))
cp = int(input("Enter "))
trestbps = int(input("Enter"))
chol = int(input("Enter"))
fbs = int(input("Enter"))
restecg = int(input("Enter"))
thalch = int(input("Enter"))
exang = int(input("Enter"))
oldpeak = int(input("Enter"))
slope  = float(input("Enter"))

input_data = (Age,Sex,cp,trestbps,chol,fbs,restecg,thalch,exang,oldpeak,slope)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model_xg.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
    print('The Person does not have a Heart Disease')
elif(prediction[0]== 1):
    print("Moderate")
elif(prediction[0]==2):
    print('Moderately high')

else:
    print('The Person has Heart Disease')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




