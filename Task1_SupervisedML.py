#!/usr/bin/env python
# coding: utf-8

# # Sparks Foundation Task1

# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.
# 
# 

# What will be predicted score if a student study for 9.25 hrs in a day?

# In[5]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# Reading data from remote link
url = "http://bit.ly/w-data"
stud_data = pd.read_csv(url)
print("Data imported successfully")

stud_data.head(10)


# In[9]:


# check for missing values
stud_data.isna().sum()


# In[11]:


stud_data.describe()


# In[12]:


# Plotting the distribution of scores
stud_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# # Preparing the Data

# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[13]:


X = stud_data.iloc[:, :-1].values
y = stud_data.iloc[:, 1].values


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[14]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# # Training the Algorithm

# We have split our data into training and testing sets, 
# and now is finally the time to train our algorithm.
# 
# 

# In[15]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[16]:


X_train


# In[18]:


y_train


# In[22]:


#Plotting the regression line
line= regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line, c='green');
plt.xlabel("No. of hours studied")
plt.ylabel(" Percentage score")
plt.show()


# # Making Predictions

# Now that we have trained our algorithm, it's time to make some predictions.

# In[23]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[24]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# Q). What will be predicted score if a student studies for 9.25 hrs/ day?

# In[28]:


hours=np.array([9.25])
prediction = regressor.predict(hours.reshape(-1,1))
print("No of Hours = {}\n".format(hours))
print("Predicted Score = {}\n".format(prediction))


# # Evaluation¶

# In[29]:


from sklearn import metrics 


# In[30]:


print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

print('Mean Squared Error:',
      metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Absolute Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[32]:


# Identifying the accuracy 
regressor.score(X,y)


# In[ ]:




