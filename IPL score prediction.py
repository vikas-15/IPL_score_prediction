#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("C:\\Users\\91701\\Desktop\\assingments\\IPL_data.csv")


# In[3]:


df.head()


# In[4]:


df.columns.unique()


# In[5]:


df["venue"].unique()


# In[6]:


df.shape


# In[7]:


df.isnull().count()


# In[8]:


df.count()


# In[9]:


columns_to_remove=["mid","batsman","bowler","striker","non-striker"]


# In[10]:


df.drop(columns_to_remove,axis=1,inplace=True)


# In[11]:


df.head()


# In[12]:


print("after removing unwanted columns :{}".format(df.shape))


# In[13]:


df.index


# In[14]:


df["bat_team"].unique()


# In[15]:


consistent_teams=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Delhi Daredevils',
        'Sunrisers Hyderabad']


# In[16]:


df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
print('After removing inconsistent teams: {}'.format(df.shape))


# In[17]:


df=df[df["overs"]>5.0]


# In[18]:


print('After removing first five over: {}'.format(df.shape))


# In[19]:


df.describe()


# In[20]:


df.hist(bins=50,figsize=(20,15))
plt.show()


# In[21]:


corr_matrix=df.corr()
corr_matrix["total"].sort_values(ascending=False)


# In[22]:


correlation=corr_matrix.index
plt.figure(figsize=(13,10))
g=sns.heatmap(data=df[correlation].corr(),annot=True,cmap='RdYlGn')


# In[23]:


from datetime import datetime
print("Before converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))
print("After converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))


# In[24]:


encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
encoded_df.columns


# In[25]:


encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]


# In[26]:


encoded_df.head()


# In[27]:


X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

Y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
Y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)
print("Training set: {} and Test set: {}".format(X_train.shape, X_test.shape))


# In[28]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline=Pipeline([
            ("Imputer",SimpleImputer(strategy="median")),
            #("Std_scaler",StandardScaler())
])


# In[29]:


X_train_tr=my_pipeline.fit_transform(X_train)


# In[30]:


X_train_tr.shape


# In[31]:


X_tr=pd.DataFrame(X_train_tr,columns=X_train.columns)


# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
#model=RandomForestRegressor()
model=LinearRegression()
#model=GaussianNB()
#model=DecisionTreeRegressor()
model.fit(X_train_tr,Y_train)


# In[33]:


X_test_tr=my_pipeline.fit_transform(X_test)
predictions=model.predict(X_test_tr)


# In[34]:


predictions


# In[35]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
final_mse=mean_squared_error(Y_test,predictions)
final_mae=mean_absolute_error(Y_test,predictions)
final_rmse=np.sqrt(final_mse)
print("Mean absolute error: {}".format((final_mae)))
print("Mean squared error: {}".format((final_mse)))
print(" Root Mean squared error: {}".format((final_rmse)))


# In[36]:


final_rmse


# In[37]:


from sklearn.ensemble import AdaBoostRegressor
adb_regressor = AdaBoostRegressor(base_estimator=model, n_estimators=100)
adb_regressor.fit(X_train_tr, Y_train)
predictions=model.predict(X_test_tr)


# In[38]:


final_mse=mean_squared_error(Y_test,predictions)
final_mae=mean_absolute_error(Y_test,predictions)
final_rmse=np.sqrt(final_mse)
print("Mean absolute error: {}".format((final_mae)))
print("Mean squared error: {}".format((final_mse)))
print(" Root Mean squared error: {}".format((final_rmse)))


# In[39]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,X_train_tr,Y_train,scoring="neg_mean_squared_error",cv=10)
rmse_score=np.sqrt(-scores)


# In[40]:


rmse_score


# In[41]:


def predict_score(batting_team, bowling_team , overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5):
  temp_array = list()

  # Batting Team
  if batting_team == 'Chennai Super Kings':
    temp_array = temp_array + [1,0,0,0,0,0,0,0]
  elif batting_team == 'Delhi Daredevils':
    temp_array = temp_array + [0,1,0,0,0,0,0,0]
  elif batting_team == 'Kings XI Punjab':
    temp_array = temp_array + [0,0,1,0,0,0,0,0]
  elif batting_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0,0,0,1,0,0,0,0]
  elif batting_team == 'Mumbai Indians':
    temp_array = temp_array + [0,0,0,0,1,0,0,0]
  elif batting_team == 'Rajasthan Royals':
    temp_array = temp_array + [0,0,0,0,0,1,0,0]
  elif batting_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0,0,0,0,0,0,1,0]
  elif batting_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0,0,0,0,0,0,0,1]

  # Bowling Team
  if bowling_team == 'Chennai Super Kings':
    temp_array = temp_array + [1,0,0,0,0,0,0,0]
  elif bowling_team == 'Delhi Daredevils':
    temp_array = temp_array + [0,1,0,0,0,0,0,0]
  elif bowling_team == 'Kings XI Punjab':
    temp_array = temp_array + [0,0,1,0,0,0,0,0]
  elif bowling_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0,0,0,1,0,0,0,0]
  elif bowling_team == 'Mumbai Indians':
    temp_array = temp_array + [0,0,0,0,1,0,0,0]
  elif bowling_team == 'Rajasthan Royals':
    temp_array = temp_array + [0,0,0,0,0,1,0,0]
  elif bowling_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0,0,0,0,0,0,1,0]
  elif bowling_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0,0,0,0,0,0,0,1]
    

  # Overs, Runs, Wickets, Runs_in_prev_5, Wickets_in_prev_5
  temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]

  # Converting into numpy array
  temp_array = np.array([temp_array])

  # Prediction
  return int(model.predict(temp_array)[0])

# Prediction 1
• Date: 14th April 2019
• IPL : Season 12
• Match number: 30
• Teams: Sunrisers Hyderabad vs. Delhi Daredevils
• First Innings final score: 155/7
# In[42]:


final_score = predict_score(batting_team='Delhi Daredevils', bowling_team='Sunrisers Hyderabad', overs=11.5, runs=98, wickets=3, runs_in_prev_5=41, wickets_in_prev_5=1)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))


# In[43]:


final_score = predict_score(batting_team='Chennai Super Kings', bowling_team='Mumbai Indians', overs=11.5, runs=98, wickets=3, runs_in_prev_5=41, wickets_in_prev_5=1)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))

# Prediction 2
• Date: 10th May 2019
• IPL : Season 12
• Match number: 59 (Eliminator)
• Teams: Delhi Daredevils vs. Chennai Super Kings
• First Innings final score: 147/9
# In[44]:


final_score = predict_score(batting_team='Delhi Daredevils', bowling_team='Chennai Super Kings', overs=10.2, runs=68, wickets=3, runs_in_prev_5=29, wickets_in_prev_5=1)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))

Prediction 3
• Date: 11th April 2019
• IPL : Season 12
• Match number: 25
• Teams: Rajasthan Royals vs. Chennai Super Kings
• First Innings final score: 151/7
# In[45]:


final_score = predict_score(batting_team='Rajasthan Royals', bowling_team='Chennai Super Kings', overs=13.3, runs=92, wickets=5, runs_in_prev_5=27, wickets_in_prev_5=2)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))

Prediction 5
• Date: 17th May 2018
• IPL : Season 11
• Match number: 50
• Teams: Mumbai Indians vs. Kings XI Punjab
• First Innings final score: 186/8
# In[46]:


final_score = predict_score(batting_team='Mumbai Indians', bowling_team='Kings XI Punjab', overs=12.2, runs=110, wickets=3, runs_in_prev_5=27, wickets_in_prev_5=1)
print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))


# In[47]:


import pickle
filename = 'IPL_score_prediction.pkl'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:




