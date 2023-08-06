#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn import linear_model
import warnings
warnings.simplefilter('ignore')


import itertools


# In[3]:


df=pd.read_csv("EarthCam_DataScience.csv")
df


# In[4]:


df.info()


# In[5]:


#data cleaning


# In[6]:


df=df.replace('',np.nan).fillna(0)
df.head()


# In[7]:


df=df.replace('[]',np.nan).fillna(0)
df.head()


# In[8]:


df.drop(['Lightning'],axis=True,inplace=True)
df


# In[34]:


df['ObservedAt_DateTime'] = pd.to_datetime(df['ObservedAt_DateTime'])
df['Month'] = df['ObservedAt_DateTime'].dt.month
df['Day'] = df['ObservedAt_DateTime'].dt.day
df['Year'] = df['ObservedAt_DateTime'].dt.year
df.drop('ObservedAt_DateTime', axis=1, inplace=True)
df.head()


# In[35]:


print("A detailed description of the dataset:")
df.describe()


# In[36]:


df.info()


# In[37]:


df[df.duplicated()]


# In[38]:


df['Station'].unique()


# In[39]:


df.sort_values(by='Temperature')


# In[40]:


df.shape


# In[41]:


plt.figure(figsize=(12,8))
g = sns.relplot(x='Temperature', y='FeelsLike', data=df, kind="scatter", color="purple")
g.fig.suptitle("Temperature (C) vs. FeelsLike",y=1.1)
plt.show()

from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


# In[42]:


plt.figure(figsize=(12,8))
g = sns.relplot(x='Dewpoint',y='FeelsLike', data=df, kind="scatter", color="cyan", ci=None)
g.fig.suptitle("Dewpoint vs. FeelsLike",y=1.1)
plt.show()

from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


# In[43]:


sns.countplot(x=df['CloudReport'])


# In[44]:


sns.countplot(x=df['ConditionType'])


# In[48]:


plt.figure(figsize=(10, 6))

# Box plot of RelativeHumidity for each day
sns.boxplot(x='Day', y='RelativeHumidity', data=df)
plt.xlabel('Day')
plt.ylabel('Relative Humidity')
plt.title('Relative Humidity by Day')


# In[50]:


plt.figure(figsize=(10, 6))

# Box plot of RelativeHumidity for each Month
sns.boxplot(x='Month', y='RelativeHumidity', data=df)
plt.xlabel('Month')
plt.ylabel('Relative Humidity')
plt.title('Relative Humidity by Month')


# In[51]:


ls=linear_model.LinearRegression()
X=df['Day'].values.reshape(-1,1)
Y=df['Temperature'].values.reshape(-1,1)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,shuffle=True,random_state=0)


lrClassifier=LogisticRegression()


# In[52]:


lrClassifier.fit(X_train,y_train)


# In[53]:


prediction = lrClassifier.predict(X_test)


# In[54]:


prediction[:5000]


# In[55]:


y_test[:500]


# In[56]:


ls=linear_model.LinearRegression()
X=df['Month'].values.reshape(-1,1)
Y=df['Temperature'].values.reshape(-1,1)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,shuffle=True,random_state=0)


lrClassifier=LogisticRegression()


# In[57]:


lrClassifier.fit(X_train,y_train)


# In[ ]:




