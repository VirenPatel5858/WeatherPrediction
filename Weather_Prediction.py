#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


weather_df = pd.read_csv("Chicago.csv", index_col="DATE")
weather_df


# ### Data cleaning:

# In[3]:


# Calculating the total number of null values in the weather dataset

weather_df.apply(pd.isnull).sum()


# In[4]:


# Calculating the percentage of missing values in each column

null_percent= weather_df.apply(pd.isnull).sum()/weather_df.shape[0]
null_percent


# In[5]:


# Selecting columns whose missing values are less than 5%

valid_columns = weather_df.columns[null_percent < 0.05]
valid_columns


# In[6]:


# Selecting the valid columns only

weather_df = weather_df[valid_columns].copy()
weather_df


# In[7]:


weather_df.apply(pd.isnull).sum()


# In[8]:


# Filling the mssing values with linear interpolation
   
weather_df = weather_df.fillna({'AWND':weather_df['AWND'].interpolate(method='linear') })
weather_df = weather_df.fillna({'PRCP':weather_df['PRCP'].interpolate(method='linear') })
weather_df = weather_df.fillna({'TMAX':weather_df['TMAX'].interpolate(method='linear') })
weather_df = weather_df.fillna({'TMIN':weather_df['TMIN'].interpolate(method='linear') })


# In[9]:


# Checking for missing values again
weather_df.apply(pd.isnull).sum()


# In[10]:


weather_df.dtypes


# In[11]:


weather_df.index


# In[12]:


# Converting index to date time

weather_df.index = pd.to_datetime(weather_df.index)
weather_df.index


# In[13]:


weather_df


# In[14]:


# Checking if there is any gap in the data 

weather_df.index.year.value_counts().sort_index()


# In[15]:


# Setting the target column in weather dataset which is Tmax

weather_df['target']=weather_df.shift(-1)['TMAX']
weather_df


# In[16]:


# Filling the last Nan 
weather_df=weather_df.ffill()


# In[17]:


weather_df


# ### Data Visualisation:

# In[18]:


# Plotting minimum and maximum temperature to check noisy data 

fig, ax = plt.subplots(figsize=(10,5))
fig.suptitle('Maximum and Minimum Temperature for the dataset',fontsize=12)

ax.plot(weather_df['TMAX'], linewidth=1.0, color = 'Blue', label = "TMAX")
ax.plot(weather_df['TMIN'], linewidth=1.0, color = 'Orange', label = "TMIN")

ax.legend()

ax.set( xlabel="Date Time",
        ylabel="Temperature (F)")

plt.show()
plt.tight_layout()


# In[19]:


# To observe the correlation of the data set 
weather_df.corr()


# In[20]:


# Data Visualization

plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(weather_df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# ### Functions for Linear Regression Models:

# In[21]:


from sklearn.linear_model import Ridge
model_1 = Ridge(alpha=0.1)

def ridge_predictor(weather_df, model, predictors, start=7667, step=90):
    
    all_predictions=[]
    
    for i in range(start, weather_df.shape[0], step):
        train = weather_df.iloc[:i,:]
        test = weather_df.iloc[i:(i+step),:]
        
        model_1.fit(train[predictors], train['target'])
        
        preds = model_1.predict(test[predictors])
        
        preds = pd.Series(preds, index=test.index)
        combined= pd.concat([test['target'], preds], axis=1)
        
        combined.columns=["actual", "prediction"]
        
        combined["difference"] = (combined["actual"]-combined["prediction"]).abs()   
        
        all_predictions.append(combined)
    
    return pd.concat(all_predictions)


# In[22]:


from sklearn.linear_model import LinearRegression
model_2 = LinearRegression()

def linear_predictor(weather_df, model, predictors, start=7667, step=90):
    
    all_predictions=[]
    
    for i in range(start, weather_df.shape[0], step):
        train = weather_df.iloc[:i,:]
        test = weather_df.iloc[i:(i+step),:]
        
        model_2.fit(train[predictors], train['target'])
        
        preds = model_2.predict(test[predictors])
        
        preds = pd.Series(preds, index=test.index)
        combined= pd.concat([test['target'], preds], axis=1)
        
        combined.columns=["actual", "prediction"]
        
        combined["difference"] = (combined["actual"]-combined["prediction"]).abs()   
        
        all_predictions.append(combined)
    
    return pd.concat(all_predictions)


# In[23]:


from sklearn.linear_model import Lasso
model_3 = Lasso(alpha=0.1)

def lasso_predictor(weather_df, model, predictors, start=7667, step=90):
    
    all_predictions=[]
    
    for i in range(start, weather_df.shape[0], step):
        train = weather_df.iloc[:i,:]
        test = weather_df.iloc[i:(i+step),:]
        
        model_3.fit(train[predictors], train['target'])
        
        preds = model_3.predict(test[predictors])
        
        preds = pd.Series(preds, index=test.index)
        combined= pd.concat([test['target'], preds], axis=1)
        
        combined.columns=["actual", "prediction"]
        
        combined["difference"] = (combined["actual"]-combined["prediction"]).abs()   
        
        all_predictions.append(combined)
    
    return pd.concat(all_predictions)


# ### Functions for data visualization after getting predictions:

# In[24]:


def lineplot(df):

    fig, ax = plt.subplots(figsize=(10,5))
    fig.suptitle('Actual vs Predicted values',fontsize=12)

    ax.plot(df['actual'],  linewidth=2.0,label = "Actual")
    ax.plot(df['prediction'], color = 'Orange', linewidth=2.0, label = "Predicted")

    ax.legend()

    ax.set( xlabel="Date Time",
            ylabel=" Maximum Temperature (F)")

    plt.show()
    plt.tight_layout()


# In[25]:


def violinplot(df):
    fig, ax = plt.subplots(figsize=(5,5))
    fig.suptitle('Violin plot for actual vs predicted values',fontsize=12)
    ax.set( ylabel=" Maximum Temperature (F)")
    sns.violinplot(df[["actual","prediction"]])


# In[26]:


def errorgraph(df):
    fig, ax = plt.subplots(figsize=(10,5))
    fig.suptitle('Error graph',fontsize=12)

    ax.plot(df["difference"].round().value_counts().sort_index(), linewidth=1.0)

    ax.set( xlabel="Error in prediction",
            ylabel="Count of values")

    plt.show()
    plt.tight_layout()


# ### Functions for calculating errors and accuracy of the model

# In[27]:


def rmse_error(actual, predicted):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    mse= mean_squared_error(actual,predicted)
    rmse = np.sqrt(mean_squared_error(actual,predicted))
    return rmse


# In[28]:


def r2_error(actual, predicted):
    from sklearn.metrics import r2_score
    r2 = r2_score(actual, predicted)
    return r2


# In[29]:


def mab_error(actual, predicted):
    from sklearn.metrics import mean_absolute_error as mae
    error = mae(actual, predicted)
    return error


# ## Ridge Regression - Model 1: ( 'AWND', 'PRCP', 'TMAX', 'TMIN' )

# In[30]:


# Generating columns as predictors for model

predictors_1 = weather_df.columns[~weather_df.columns.isin(["target","STATION","NAME"])]
predictors_1


# In[31]:


# Calling the ride_predictor function to get the predicted values

predictions_1 = ridge_predictor(weather_df, model_1, predictors_1)
predictions_1


# In[32]:


lineplot(predictions_1) # calling lineplot() function created 


# In[33]:


violinplot(predictions_1) # calling violinplot() function created


# In[34]:


errorgraph(predictions_1)


# In[35]:


error = rmse_error(predictions_1['actual'],predictions_1['prediction'])
print("Root mean square error : ", error)


# In[36]:


error = r2_error(predictions_1['actual'],predictions_1['prediction'])
print("R2 score error : ", error)


# In[37]:


error = mab_error(predictions_1['actual'],predictions_1['prediction'])
print("Mean absolute error : ", error)


# ###  Rolling mean function add columns to improve accuracy

# In[38]:


# rolling() function calculates the rolling mean for the number of days mentioned

def rolling(weather_df, days, col):
    new_col = f"Rolling_{days}_{col}"
    weather_df[new_col] = weather_df[col].rolling(days).mean()
    return weather_df

rolling_days = [5,15]

for days in rolling_days:
    for col in ["PRCP","TMAX","TMIN"]:
        weather_df = rolling(weather_df, days, col)


# In[39]:


weather_df


# In[40]:


# Starting values have Nan since it does not have past values to take rolling mean, So we need to cut out the first 15 rows

weather_df = weather_df.iloc[15:,:]


# In[41]:


weather_df


# In[42]:


weather_df.isna().sum()


# In[43]:


weather_df.columns


# In[44]:


# Data Visualization (Correlation heatmap)

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(weather_df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# ## Ridge Regression - Model 2 :
# 
# ('AWND', 'PRCP', 'TMAX', 'TMIN', 'Rolling_5_PRCP', 'Rolling_5_TMAX','Rolling_5_TMIN', 'Rolling_15_PRCP', 'Rolling_15_TMAX', 'Rolling_15_TMIN')

# In[45]:


predictors_2 = weather_df.columns[~weather_df.columns.isin(["target","STATION","NAME"])]
predictors_2


# In[46]:


predictions_2 = ridge_predictor(weather_df, model_1, predictors_2)
predictions_2


# In[47]:


lineplot(predictions_2)


# In[48]:


violinplot(predictions_2)


# In[49]:


errorgraph(predictions_2)


# In[50]:


error = rmse_error(predictions_2['actual'],predictions_2['prediction'])
print("Root mean square error : ", error)


# In[51]:


error = r2_error(predictions_2['actual'],predictions_2['prediction'])
print("R2 score  : ", error)


# In[52]:


error = mab_error(predictions_2['actual'],predictions_2['prediction'])
print("Mean absolute error : ", error)


# ### Expanding mean function to add more columns

# In[53]:


def expand_mean(df):
    return df.expanding(1).mean()

for col in ["PRCP", "TMAX", "TMIN"]:
    weather_df [f"Month_Avg_{col}"] = weather_df[col].groupby(weather_df.index.month, group_keys= False).apply(expand_mean)
    weather_df [f"Day_Avg_{col}"] = weather_df[col].groupby(weather_df.index.day_of_year, group_keys= False).apply(expand_mean)


# In[54]:


weather_df


# In[55]:


# Data Visualization using heatmap

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(weather_df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# ## Ridge Regression - Model 3: 
# ('AWND', 'PRCP', 'TMAX', 'TMIN', 'Rolling_5_PRCP', 'Rolling_5_TMAX','Rolling_5_TMIN', 'Rolling_15_PRCP', 'Rolling_15_TMAX',
#  'Rolling_15_TMIN', 'Month_Avg_PRCP', 'Day_Avg_PRCP', 'Month_Avg_TMAX','Day_Avg_TMAX', 'Month_Avg_TMIN', 'Day_Avg_TMIN')

# In[56]:


predictors_3 = weather_df.columns[~weather_df.columns.isin(["target","STATION","NAME"])]
predictors_3


# In[57]:


predictions_3 = ridge_predictor(weather_df, model_1, predictors_3)
predictions_3


# In[58]:


lineplot(predictions_3)


# In[59]:


violinplot(predictions_3)


# In[60]:


errorgraph(predictions_3)


# In[61]:


error = rmse_error(predictions_3['actual'],predictions_3['prediction'])
print("Root mean square error : ", error)


# In[62]:


error = r2_error(predictions_3['actual'],predictions_3['prediction'])
print("R2 score  : ", error)


# In[63]:


error = mab_error(predictions_3['actual'],predictions_3['prediction'])
print("Mean absolute error : ", error)


# ## Linear Regression Model 1: ('AWND', 'PRCP', 'TMAX', 'TMIN')
# 

# In[64]:


# Having a look at our predictors
predictors_1


# In[65]:


# Calling the linear_predictor function to get the predicted values

predictions_1_linear = linear_predictor(weather_df, model_2, predictors_1)
predictions_1_linear


# In[66]:


lineplot(predictions_1_linear)


# In[67]:


violinplot(predictions_1_linear)


# In[68]:


errorgraph(predictions_1_linear)


# In[69]:


# Calculating the model performance

error = rmse_error(predictions_1_linear['actual'],predictions_1_linear['prediction'])
print("Root mean square error : ", error)


# In[70]:


error = r2_error(predictions_1_linear['actual'],predictions_1_linear['prediction'])
print("R2 score: ", error)


# In[71]:


error = mab_error(predictions_1_linear['actual'],predictions_1_linear['prediction'])
print("Mean absolute: ", error)


# ## Linear Regression Model 2: 
#        ('AWND', 'PRCP', 'TMAX', 'TMIN', 'Rolling_5_PRCP', 'Rolling_5_TMAX',
#        'Rolling_5_TMIN', 'Rolling_15_PRCP', 'Rolling_15_TMAX',
#        'Rolling_15_TMIN' )

# In[72]:


# Looking at our next set of predictors 
predictors_2


# In[73]:


# Calling the linear_predictor function to get the predicted values

predictions_2_linear = linear_predictor(weather_df, model_2, predictors_2)
predictions_2_linear


# In[74]:


lineplot(predictions_2_linear)


# In[75]:


violinplot(predictions_2_linear)


# In[76]:


errorgraph(predictions_2_linear)


# In[77]:


# Calculating the model performance

error = rmse_error(predictions_2_linear['actual'],predictions_2_linear['prediction'])
print("Root mean square error : ", error)


# In[78]:


error = r2_error(predictions_2_linear['actual'],predictions_2_linear['prediction'])
print("R2 score: ", error)


# In[79]:


error = mab_error(predictions_2_linear['actual'],predictions_2_linear['prediction'])
print("Mean absolute: ", error)


# ## Linear Regression Model 3: 
#      ( 'AWND', 'PRCP', 'TMAX', 'TMIN', 'Rolling_5_PRCP', 'Rolling_5_TMAX',
#        'Rolling_5_TMIN', 'Rolling_15_PRCP', 'Rolling_15_TMAX',
#        'Rolling_15_TMIN', 'Month_Avg_PRCP', 'Day_Avg_PRCP', 'Month_Avg_TMAX',
#        'Day_Avg_TMAX', 'Month_Avg_TMIN', 'Day_Avg_TMIN' )

# In[80]:


# Looking at our next set of predictors 
predictors_3 


# In[81]:


# Calling the linear_predictor function to get the predicted values

predictions_3_linear = linear_predictor(weather_df, model_2, predictors_3)
predictions_3_linear


# In[82]:


lineplot(predictions_3_linear)


# In[83]:


violinplot(predictions_3_linear)


# In[84]:


errorgraph(predictions_3_linear)


# In[85]:


# Calculating the model performance

error = rmse_error(predictions_3_linear['actual'],predictions_3_linear['prediction'])
print("Root mean square error : ", error)


# In[86]:


error = r2_error(predictions_3_linear['actual'],predictions_3_linear['prediction'])
print("R2 score: ", error)


# In[87]:


error = mab_error(predictions_3_linear['actual'],predictions_3_linear['prediction'])
print("Mean absolute: ", error)


# ## Lasso Regression Model 1: ('AWND', 'PRCP', 'TMAX', 'TMIN')

# In[88]:


# Looking at the predictors
predictors_1


# In[89]:


# Calling the lasso_predictor function to get the predicted values

predictions_1_lasso = lasso_predictor(weather_df, model_3, predictors_1)
predictions_1_lasso


# In[90]:


lineplot(predictions_1_lasso)


# In[91]:


violinplot(predictions_1_lasso)


# In[92]:


errorgraph(predictions_1_lasso)


# In[93]:


# Calculating the model performance

error = rmse_error(predictions_1_lasso['actual'],predictions_1_lasso['prediction'])
print("Root mean square error : ", error)


# In[94]:


error = r2_error(predictions_1_lasso['actual'],predictions_1_lasso['prediction'])
print("R2 score: ", error)


# In[95]:


error = mab_error(predictions_1_lasso['actual'],predictions_1_lasso['prediction'])
print("Mean absolute: ", error)


# ## Lasso Regression Model 2: 
#     ('AWND', 'PRCP', 'TMAX', 'TMIN', 'Rolling_5_PRCP', 'Rolling_5_TMAX',
#        'Rolling_5_TMIN', 'Rolling_15_PRCP', 'Rolling_15_TMAX',
#        'Rolling_15_TMIN')
# 

# In[96]:


# Looking at the predictors 
predictors_2


# In[97]:


# Calling the lasso_predictor function to get the predicted values

predictions_2_lasso = lasso_predictor(weather_df, model_3, predictors_2)
predictions_2_lasso


# In[98]:


lineplot(predictions_2_lasso)


# In[99]:


violinplot(predictions_2_lasso)


# In[100]:


errorgraph(predictions_2_lasso)


# In[101]:


# Calculating the model performance

error = rmse_error(predictions_2_lasso['actual'],predictions_2_lasso['prediction'])
print("Root mean square error : ", error)


# In[102]:


error = r2_error(predictions_2_lasso['actual'],predictions_2_lasso['prediction'])
print("R2 score: ", error)


# In[103]:


error = mab_error(predictions_2_lasso['actual'],predictions_2_lasso['prediction'])
print("Mean absolute: ", error)


# ## Lasso Regression Model 3:
#     ('AWND','PRCP','TMAX','TMIN', 'Rolling_5_PRCP','Rolling_5_TMAX','Rolling_5_TMIN','Rolling_15_PRCP','Rolling_15_TMAX',           'Rolling_15_TMIN','Month_Avg_PRCP','Day_Avg_PRCP','Month_Avg_TMAX','Day_Avg_TMAX','Month_Avg_TMIN','Day_Avg_TMIN')

# In[104]:


# Looking at the predictors 
predictors_3


# In[105]:


# Calling the lasso_predictor function to get the predicted values

predictions_3_lasso = lasso_predictor(weather_df, model_3, predictors_3)
predictions_3_lasso


# In[106]:


lineplot(predictions_3_lasso)


# In[107]:


violinplot(predictions_3_lasso)


# In[108]:


errorgraph(predictions_3_lasso)


# In[109]:


# Calculating the model performance

error = rmse_error(predictions_3_lasso['actual'],predictions_3_lasso['prediction'])
print("Root mean square error : ", error)


# In[110]:


error = r2_error(predictions_3_lasso['actual'],predictions_3_lasso['prediction'])
print("R2 score: ", error)


# In[111]:


error = mab_error(predictions_3_lasso['actual'],predictions_3_lasso['prediction'])
print("Mean absolute: ", error)


# ### Splitting the dataset for the next 2 models:

# In[112]:


train_dataset = weather_df.loc["2000-01-01" : "2020-12-31"] 


# In[113]:


test_dataset = weather_df.loc["2021-01-01":]


# In[114]:


x_train = train_dataset[['AWND', 'PRCP', 'TMAX', 'TMIN', 'Rolling_5_PRCP', 'Rolling_5_TMAX','Rolling_5_TMIN', 'Rolling_15_PRCP', 'Rolling_15_TMAX', 'Rolling_15_TMIN', 'Month_Avg_PRCP', 'Day_Avg_PRCP', 'Month_Avg_TMAX','Day_Avg_TMAX', 'Month_Avg_TMIN', 'Day_Avg_TMIN']]
x_test = test_dataset[['AWND', 'PRCP', 'TMAX', 'TMIN', 'Rolling_5_PRCP', 'Rolling_5_TMAX','Rolling_5_TMIN', 'Rolling_15_PRCP', 'Rolling_15_TMAX', 'Rolling_15_TMIN', 'Month_Avg_PRCP', 'Day_Avg_PRCP', 'Month_Avg_TMAX','Day_Avg_TMAX', 'Month_Avg_TMIN', 'Day_Avg_TMIN']]


# In[115]:


y_train = train_dataset[['target']]
y_test = test_dataset[['target']]


# ### Elastic Regression Model: 
#  ('AWND', 'PRCP', 'TMAX', 'TMIN', 'Rolling_5_PRCP', 'Rolling_5_TMAX','Rolling_5_TMIN', 'Rolling_15_PRCP', \'Rolling_15_TMAX', 'Rolling_15_TMIN', 'Month_Avg_PRCP', 'Day_Avg_PRCP', 'Month_Avg_TMAX','Day_Avg_TMAX', 'Month_Avg_TMIN', 'Day_Avg_TMIN')

# In[116]:


from sklearn.linear_model import ElasticNet 
enet = ElasticNet(alpha =1.0, l1_ratio = 0.5)


# In[117]:


enet.fit(x_train, y_train)


# In[118]:


y_pred = enet.predict(x_test)


# In[119]:


preds = pd.DataFrame(y_pred, index=x_test.index, columns = ['prediction'])


# In[120]:


preds = pd.concat([y_test['target'],preds['prediction']], axis=1)
preds.columns=["actual", "prediction"]
preds


# In[121]:


# Calculating difference column in dataset

for i in preds:
    preds['difference']=(preds["actual"]-preds["prediction"]).abs()

preds


# In[122]:


lineplot(preds)


# In[123]:


violinplot(preds)


# In[124]:


errorgraph(preds)


# In[125]:


error = rmse_error(preds['actual'],preds['prediction'])
print("Root mean square error : ", error)


# In[126]:


error = r2_error(preds['actual'],preds['prediction'])
print("R2 score: ", error)


# In[127]:


error = mab_error(preds['actual'],preds['prediction'])
print("Mean absolute: ", error)


# ## Random Forest Regressor Model : 
# ('AWND', 'PRCP', 'TMAX', 'TMIN', 'Rolling_5_PRCP', 'Rolling_5_TMAX','Rolling_5_TMIN', 'Rolling_15_PRCP', 'Rolling_15_TMAX', 'Rolling_15_TMIN', 'Month_Avg_PRCP', 'Day_Avg_PRCP', 'Month_Avg_TMAX','Day_Avg_TMAX', 'Month_Avg_TMIN', 'Day_Avg_TMIN')

# In[128]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators= 200 , random_state = 50)
rf.fit(x_train, y_train)


# In[129]:


y_pred_rf = rf.predict(x_test)


# In[130]:


preds_rf = pd.DataFrame(y_pred_rf, index=x_test.index, columns = ['prediction'])
preds_rf = pd.concat([y_test['target'],preds_rf['prediction']], axis=1)
preds_rf.columns=["actual", "prediction"]
preds_rf


# In[131]:


# Calculating difference column in dataset

for i in preds:
    preds_rf['difference']=(preds_rf["actual"]-preds_rf["prediction"]).abs()

preds_rf


# In[132]:


lineplot(preds_rf)


# In[133]:


violinplot(preds_rf)


# In[134]:


errorgraph(preds_rf)


# In[135]:


error = rmse_error(preds_rf['actual'],preds_rf['prediction'])
print("Root mean square error : ", error)


# In[136]:


error = r2_error(preds_rf['actual'],preds_rf['prediction'])
print("R2 score: ", error)


# In[137]:


error = mab_error(preds_rf['actual'],preds_rf['prediction'])
print("Mean absolute: ", error)


# In[ ]:




