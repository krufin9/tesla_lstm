#!/usr/bin/env python
# coding: utf-8

# # Tesla Stock Prediction using LSTM Model

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Getting the Price values of Tesla from last 7 years.

# In[2]:


tesla = yf.download(tickers='TSLA', period='1y', interval='1d')


# In[3]:


tesla = tesla.reset_index()['Close']


# Let's plot the stock price values of Tesla.

# In[4]:


tesla_plot = px.line(tesla, y="Close",width=1000, height=650)
tesla_plot.show()


# Data Normalization: Data standardization is the concept of rescaling the values so that they have mean as 0, and variance as 1. The aim behind this is to have all features as common values so that there is no distortion in the range of values. MinMaxScaler() is used for this. We are setting the minmaxscaler range to 0,1 with the function MinMaxScaler(). So all the values of the features will be reshaped into the range of 0 to 1.

# In[5]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
tesla = scaler.fit_transform(np.array(tesla).reshape(-1,1))


# Train-Test Split

# In[6]:


training_data_size = int(len(tesla)*0.65)
test_data_size = len(tesla)-training_data_size
train_data = tesla[0:training_data_size,:]
test_data = tesla[training_data_size:len(tesla),:1]


# In[7]:


training_data_size, test_data_size


# Converting an array of values into a dataset matrix.

# In[8]:


import numpy
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]    
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# Reshaping into X=t,t+1,t+2,t+3 (independent values) and Y=t+4.

# In[9]:


time_step = 50                                          ###.    x_train,y_train        x_test,y_test
X_train, y_train = create_dataset(train_data, time_step) ###i=0, 0,1,2,3-----49         50
X_test, y_test = create_dataset(test_data, time_step)


# In[10]:


X_train.shape, y_train.shape


# In[11]:


X_test.shape, y_test.shape


# Reshaping for LSTM Model input to be [samples, time steps, features].

# In[12]:


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# Stacked LSTM model

# In[13]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[14]:


from keras.layers import Dropout


# In[15]:


model_tesla = Sequential()
model_tesla.add(LSTM(50,return_sequences=True,input_shape=(50,1)))
model_tesla.add(Dropout(0.2))
model_tesla.add(LSTM(50,return_sequences=True))
model_tesla.add(Dropout(0.2))
model_tesla.add(LSTM(50))
model_tesla.add(Dropout(0.2))
model_tesla.add(Dense(1))
model_tesla.compile(loss='mean_squared_error',optimizer='adam')


# In[16]:


model_tesla.summary()


# In[17]:


model_tesla.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=250,batch_size=32,verbose=1)


# In[18]:


import tensorflow as tf


# Prediction and checking the Performance matrix

# In[19]:


train_predict = model_tesla.predict(X_train)
test_predict = model_tesla.predict(X_test)


# Inverse Transform to original form.

# In[20]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# Calculating RMSE (Root mean squared error) of y_train and y_test

# In[21]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[22]:


math.sqrt(mean_squared_error(y_test,test_predict))


# Shifting the train and test predictions for plotting.

# In[23]:


look_back = 50
trainPredictPlot = np.empty_like(tesla)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(tesla)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(tesla)-1, :] = test_predict

plt.figure(figsize=(16,10))
plt.plot(scaler.inverse_transform(tesla))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[24]:


len(test_data)


# In[25]:


x_input=test_data[39:].reshape(1,-1)
x_input.shape


# In[26]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[27]:


temp_input[:5]


# For prediction of next 10 days.

# In[28]:


from numpy import array

lst_output=[]
n_steps=50
i=0
while(i<10):
    
    if(len(temp_input)>50):
        
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model_tesla.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model_tesla.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output[:10])


# In[29]:


day_new = np.arange(1,51)
day_pred = np.arange(51,61)


# In[30]:


len(tesla)


# Plotting the FINAL results

# In[31]:


plt.plot(day_new,scaler.inverse_transform(tesla[202:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[32]:


new_tesla = tesla.tolist()
new_tesla.extend(lst_output)
plt.plot(new_tesla[200:])


# In[33]:


new_tesla=scaler.inverse_transform(new_tesla).tolist()
plt.plot(new_tesla)

