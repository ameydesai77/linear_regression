#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression

# In[1]:


#Let's start with importing necessary libraries

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Let's create a function to create adjusted R-Squared
def adj_r2(x,y):
    r2 = LR.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[3]:


data =pd.read_csv('C:\\Users\AmEy\personal\ML_Series\DATASETS\Admission_Prediction.csv')
data.head()


# In[4]:


data.describe(include='all')


# In[5]:


data.isnull().sum()/len(data)*100


# In[6]:


data['University Rating'].value_counts()


# In[7]:


data['University Rating'].mode()


# In[8]:


data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])
data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
data['GRE Score']  = data['GRE Score'].fillna(data['GRE Score'].mean())


# In[9]:


data.describe()

Now the data looks good and there are no missing values. Also, the first cloumn is just serial numbers, so we don' need that column. Let's drop it from data and make it more clean.
# In[10]:


data= data.drop(['Serial No.'],axis=1)
data.head()

data= data.drop(columns = ['Serial No.'])
data.head()Let's visualize the data and analyze the relationship between independent and dependent variables:
# In[11]:


# let's see how data is distributed for every column(UNIVARITE ANALYSIS)
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=20 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.tight_layout()

The data distribution looks decent enough and there doesn't seem to be any skewness. Great let's go ahead!

Let's observe the relationship between independent variables and dependent variable.
# In[12]:


X =data.drop(columns = ['Chance of Admit'])
y = data['Chance of Admit']


# In[13]:


#(BIVARIATE ANALYSIS)
plt.figure(figsize=(20,30), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=15 :
        ax = plt.subplot(5,3,plotnumber)
        plt.scatter(X[column],y)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Chance of Admit',fontsize=20)
    plotnumber+=1
plt.tight_layout()

Great, the relationship between the dependent and independent variables look fairly linear. Thus, our linearity assumption is satisfied.

Let's move ahead and check for multicollinearity.
# In[14]:


scaler =StandardScaler()

X_scaled = scaler.fit_transform(X)


# In[15]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
# we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = X.columns


# In[16]:


vif

Here, we have the correlation values for all the features. As a thumb rule, a VIF value greater than 5 means a very severe multicollinearity. We don't any VIF greater than 5 , so we are good to go.
Great. Let's go ahead and use linear regression and see how good it fits our data. But first. let's split our data in train and test.
# In[17]:


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)


# In[18]:


y_train.head()


# In[19]:


LR = LinearRegression()

LR.fit(x_train,y_train)


# In[20]:


y_pred =LR.predict(x_test)


# In[21]:


R2_score=LR.score(x_train,y_train)


# In[22]:


print('R2 score for train data is :  ',R2_score)

print('Adj_R2 score for train data is :  ',adj_r2(x_train,y_train))

Our r2 score is 84.15% and adj r2 is 83.85% for our training et., so looks like we are not being penalized by use of any feature.
Let's check how well model fits the test data.

# In[23]:


LR.score(x_test,y_test) 

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
score ----0.7534898831471069
# In[24]:


print('Adjusted r2 for train data: ' ,adj_r2(x_train,y_train))
print('Adjusted r2 for test data: ',adj_r2(x_test,y_test))

##### here adj_r2  for train is greater than adj_r2 for test indicates the model is overfitted
# Now let's check if our model is overfitting our data using regularization.#
# In[25]:


# saving the model to the local file system
filename = 'finalized_model.pickle'
pickle.dump(LR, open(filename, 'wb'))


# In[26]:


# prediction using the saved model
loaded_model = pickle.load(open(filename, 'rb'))
a=loaded_model.predict(scaler.transform([[300,110,5,5,5,10,1]]))
a


# In[27]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[28]:


from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor # since target variable is continuous in nature

rfr =RandomForestRegressor(n_estimators=10)

rfr.fit(x_train,y_train)
#print(accuracy_score(y_test, y_pred))


# In[29]:


rfr.score(x_test,y_test)


# In[30]:


y_pred1=LR.predict(scaler.transform([[300,110,5,5,5,10,1]]))


# In[31]:


y_pred1


# In[32]:


# Lasso Regularization
# LassoCV will return best alpha and coefficients after performing 10 cross validations
lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
lasscv.fit(x_train, y_train)


# In[33]:


# best alpha parameter
alpha = lasscv.alpha_
alpha


# In[34]:


#now that we have best parameter, let's use Lasso regression and see how well our data has fitted before

lasso_reg = Lasso(alpha)
lasso_reg.fit(x_train, y_train)


# In[35]:


lasso_reg.score(x_test, y_test)

our r2_score for test data (75.34%) comes same as before using regularization. So, it is fair to say our OLS model did not overfit the data.
# In[36]:


# Using Ridge regression model
# RidgeCV will return best alpha and coefficients after performing 10 cross validations. 
# We will pass an array of random numbers for ridgeCV to select best alpha from them

alphas = np.random.uniform(low=0, high=10, size=(50,))
ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
ridgecv.fit(x_train, y_train)


# In[37]:


alpha = ridgecv.alpha_
alpha


# In[38]:


ridge_reg = Ridge(alpha)
ridge_reg.fit(x_train, y_train)


# In[39]:


ridge_reg.score(x_test, y_test)

we got the same r2 square using Ridge regression as well. So, it's safe to say there is no overfitting.
# In[40]:


# Elastic net

elasticCV = ElasticNetCV(alphas = None, cv =10)

elasticCV.fit(x_train, y_train)


# In[41]:


alpha=elasticCV.alpha_


# In[42]:


# l1_ratio gives how close the model is to L1 regularization, below value indicates we are giving equal
#preference to L1 and L2
elasticCV.l1_ratio


# In[43]:


elasticnet_reg = ElasticNet(alpha = alpha,l1_ratio=0.5)
elasticnet_reg.fit(x_train, y_train)


# In[44]:


elasticnet_reg.score(x_test, y_test)

So, we can see by using different type of regularization, we still are getting the same r2 score. That means our OLS model has been well trained over the training data and there is no overfitting.