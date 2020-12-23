#!/usr/bin/env python
# coding: utf-8
Regression Model - use Air passenger dataset
1. Build a linear regression model and estimate the expected passengers for a Promotion_Budget is 650,000. 
2. Build a multiple regression model to predict the number of passengers
The objective of this model is to predict the number of Air passengers for a special Promotion_Budget,
here since the Target variable is continuous  is nature we use Regression method and since we are using 
all feature columns for prediction ,we use Regression model with multiple features
# #### first let us import the required libraries 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# #### importing the dataset from the localSource

# In[2]:


data = pd.read_csv('...\Downloads\AirPassengers.csv')


# ##### let us check the column contents by choosing first few rows ,by default  first 5 rows

# In[3]:


data.head()


# In[4]:


print('No of Rows:',data.shape[0])
print('No of Columns:',data.shape[1])


# In[5]:


data.info()


# In[6]:


cat=[]
num=[]
for i in data.columns:
    if data[i].dtypes=='object':
        cat.append(i)
    else:
        num.append(i)
print('Categorical features are:    ',   cat)
print('Numerical features are : ',num)


# In[7]:


data[num].describe().T


# In[8]:


data[cat].describe()


# In[9]:


for i in cat:
    print(i.upper())
    print(data[i].value_counts(),'\n')


# Here we can observe that almost all categorical variables consist of columns with 'Yes' and 'No' ,makes it easy for us 
# for converting them interms of 1 &  0

# In[10]:


data = data.replace(["YES","NO"],[1,0])
data.head()


# #### check for null values in all columns

# In[11]:


data.isnull().sum()


# #### There are no null values

# #### let us check whether the feature columns are corrlated by checking correlation diagramatically

# In[12]:


plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True)


# #### Here the features which are highly correlated are ommited from dataset

# In[13]:


data.drop(columns = ['Week_num','Service_Quality_Score','Bad_Weather_Ind'],axis=1,inplace=True)


# In[14]:


data


# #### Identify feature and target columns 

# In[15]:


X = data.iloc[:,1:].values
X.shape


# In[16]:


y = data.iloc[:,0].values


# #### Split the data into Train and Test 

# In[17]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# #### Now lets Scale the feature columns using Standard Scalar as the magnitude is not uniform

# In[18]:


SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)
X_test


# #### Since Target is to predict the number of passangers which is continuous in number we use Linear Regression model

# In[19]:


Lin = LinearRegression()

Lin.fit(X_train,y_train)


# In[20]:


y_pred = Lin.predict(X_test)


# In[21]:


print('Model train Score is:\n ',Lin.score(X_train,y_train))


# In[22]:


print('Model test Score is:\n ',metrics.r2_score(y_test,y_pred) )


# #### It is a balanced prediction model

# In[23]:


Lin.coef_


# In[24]:


Lin.intercept_


# #### Saving the model to the local file system

# In[25]:


import pickle
filename = 'finalized_model.pickle'
pickle.dump(Lin, open(filename, 'wb'))


# In[26]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[27]:


loaded_model = pickle.load(open(filename, 'rb'))
a=loaded_model.predict(SC.transform([[1108254,1,1,0.90,0]]))
a


# In[28]:


Lin.predict([[0.10239297, -0.57735027, -0.96490128, -0.58297314, -0.93094934]])


# In[29]:



x=[[638330,0,0,0.90,0]]
X = SC.transform(x)
X


# In[30]:


Lin.predict([[0.10358557, -0.57735027, -0.96490128,  1.41724979, -0.93094934]])


# In[ ]:





# In[ ]:





# In[ ]:




