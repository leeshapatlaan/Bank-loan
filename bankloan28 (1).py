#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
import os
import calendar 


# ''' EDA columns / null / duplicates / outliers / treatement / dummies / X,y / split,train,test '''

# In[2]:


df=pd.read_excel("bank.xlsx")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


df.columns


# In[8]:


#df["day"] = pd.to_datetime(df["day"])


# In[9]:


#df["month"] = pd.to_(df["month"])


# In[10]:


df.describe(percentiles = [0.01,0.02,0.03,0.04,0.05,.06,.07,.08,.09,.91,.92,.93,.94,.95,.96,.97,.98,.99]).T


# In[11]:


df.info()


# In[12]:


plt.boxplot(df["balance"])


# In[13]:


df["balance"]=np.where(df["balance"]>13226.98,13226.98,df["balance"] ) 
df["balance"]=np.where(df["balance"]<=0,1000,df["balance"] ) 


# In[14]:


plt.boxplot(df["balance"])


# In[15]:


plt.boxplot(df["duration"])


# In[ ]:





# In[16]:


df["duration"]=np.where(df["duration"]>1577.17,1577.17,df["duration"] ) 


# In[17]:


plt.boxplot(df["duration"])


# In[18]:


plt.boxplot(df["campaign"])


# In[19]:


df["campaign"]=np.where(df["campaign"]>13,13,df["campaign"] ) 


# In[20]:


plt.boxplot(df["campaign"])


# In[21]:


plt.boxplot(df["pdays"])


# In[22]:


df["pdays"]=np.where(df["pdays"]>425.39,425.39,df["pdays"] ) 


# In[23]:


plt.boxplot(df["pdays"])


# In[24]:


plt.boxplot(df["previous"])


# In[25]:


df["previous"]=np.where(df["previous"]>10.00,10.00,df["previous"] )


# In[26]:


plt.boxplot(df["previous"])


# In[ ]:





# In[ ]:





# In[27]:


#label encoder = when the dependent variable in yes (1)no(0) form 


# In[28]:


#dummies  = convert object into binary


# In[29]:


from sklearn.preprocessing import LabelEncoder


# In[30]:


df["default"] = LabelEncoder().fit_transform(df.deposit)


# In[31]:


df


# In[32]:


df1 = pd.get_dummies(df,
                columns = ["job","marital","education","housing","loan","contact","month","poutcome","deposit"], 
            drop_first = True)


# In[33]:


df1


# In[34]:


y=df1["default"]
X=df1.drop(columns=["default"])


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train , X_test, y_train , y_test = train_test_split (X,y, test_size = 0.3 , random_state=99) 


# In[37]:


X_train


# In[ ]:





# In[38]:


log=LogisticRegression(random_state = 99)


# In[39]:


log.fit(X_train,y_train) 


# In[40]:


print("Train Accuracy" , log.score(X_train , y_train))


# In[41]:


print("Test Accuracy" , log.score(X_test , y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




