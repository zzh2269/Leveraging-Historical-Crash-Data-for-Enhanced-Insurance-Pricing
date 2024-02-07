#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.options.display.max_columns = 100
pd.options.mode.chained_assignment = None
import joblib
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import re
import gc

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from sklearn.linear_model import  RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


#To ignore the convergence warnings
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import warnings
warnings.filterwarnings('ignore')


# In[3]:


accident_df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_March23.csv")


# In[118]:


# accident_df= accident_df[accident_df['State']=='OH']


# In[4]:


accident_df.head()


# In[5]:


accident_df.shape


# In[6]:


accident_df.dtypes


# In[7]:


accident_df.info(verbose=False, memory_usage="deep")


# In[8]:


accident_df['Severity'].value_counts().max() / len(accident_df)


# In[9]:


accident_df.isna().sum()


# In[10]:


accident_df.info()


# In[11]:


accident_df.describe()


# In[12]:


accident_df.columns


# In[129]:


# 'Source' 
#No Number, Side


# In[13]:


accident_df = accident_df.drop(['ID', 'Source', 'Description','End_Lat','End_Lng','Country',
                  'Turning_Loop','Wind_Chill(F)','Weather_Timestamp',
              'Street','Zipcode',
                 'Airport_Code'], axis=1)


# In[14]:


##Missing values
accident_df['City'] = accident_df['City'].fillna('Missing')
accident_df['Weather_Condition'] = accident_df['Weather_Condition'].fillna('Missing')
accident_df['Wind_Direction'] = accident_df['Wind_Direction'].fillna('Missing')
accident_df['Precipitation(in)'] = accident_df['Precipitation(in)'].fillna(0)

accident_df = accident_df.dropna(subset=['Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight'])


# In[15]:


accident_df


# In[16]:


# accident_df['Start_Time'] = accident_df['Start_Time'].astype('timedelta64[m]')
# accident_df['End_Time'] = accident_df['End_Time'].astype('timedelta64[m]')

accident_df['Start_Time'] = pd.to_datetime(accident_df['Start_Time'], format='mixed')
accident_df['End_Time'] = pd.to_datetime(accident_df['End_Time'], format='mixed')
    
accident_df['Year'] = accident_df['Start_Time'].dt.year
accident_df['Month'] = accident_df['Start_Time'].dt.month
accident_df['Day'] = accident_df['Start_Time'].dt.day
accident_df['Hour'] = accident_df['Start_Time'].dt.hour

accident_df['DayofWeek'] = accident_df['Start_Time'].dt.dayofweek   
dayofweek_dict = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
accident_df['DayofWeek'] = accident_df['DayofWeek'].map(dayofweek_dict)

accident_df['Time_Diff'] = (accident_df['End_Time'] - accident_df['Start_Time'])

accident_df = accident_df.drop(['Start_Time','End_Time'], axis=1)


# In[17]:


def remove_outliers(df,col):

    cutoff = df[col].std()*3
    lower_limit = df[col].mean() - cutoff
    upper_limit = df[col].mean() + cutoff
    
    df = df[(df[col] > lower_limit) & (df[col] < upper_limit)]
    return df


# In[18]:


def impute_missing_mean(df,col):

    df[col] = df[col].fillna(df[col].mean())
    return df


# In[19]:


def impute_missing_median(df,col):

    df[col] = df[col].fillna(df[col].median())
    return df


# In[20]:


accident_df['Wind_Direction'] = accident_df['Wind_Direction'].str.upper()
accident_df['Wind_Direction'] = accident_df['Wind_Direction'].str.replace('SOUTH','S')
accident_df['Wind_Direction'] = accident_df['Wind_Direction'].str.replace('WEST','W')
accident_df['Wind_Direction'] = accident_df['Wind_Direction'].str.replace('NORTH','N')
accident_df['Wind_Direction'] = accident_df['Wind_Direction'].str.replace('EAST','E')


# In[21]:


top_weather_conditions = accident_df['Weather_Condition'].value_counts().head(15).index
accident_df['Weather_Condition'] = np.where(accident_df['Weather_Condition'].isin(top_weather_conditions),
                                   accident_df['Weather_Condition'], 'Other')


# In[22]:


def bar(x,y,color,title,xlabel,ylabel):
    fig = px.bar(x=x, y=y, color=color)
    
    fig.update_layout(
    title = title,
    xaxis_title=xlabel,
    yaxis_title=ylabel)
    
    fig.show()


# In[23]:



accident_df['Time_Diff'] = accident_df['Time_Diff'].dt.total_seconds() / 60


# In[24]:


cols =['Time_Diff','Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)']
for col in cols:
    c = accident_df.shape[0]
    accident_df =remove_outliers(accident_df,col)
    n = accident_df.shape[0]
    print("{} new outliers detected for {} and removed".format(c-n,col))


# In[25]:


accident_df = impute_missing_median(accident_df,'Wind_Speed(mph)')


# In[26]:


high_severity_ = accident_df[(accident_df['Severity']==3) | (accident_df['Severity']==4) ]


# In[27]:


fig = px.pie(accident_df, values='Severity', names='Severity', title='Distribution of Accidents by Severity')
fig.show()


# In[29]:


bar(x=accident_df.groupby('State')['State'].count().sort_values(ascending=False).head(20).index,
    y=accident_df.groupby('State')['State'].count().sort_values(ascending=False).head(20).values,
   color = accident_df.groupby('State')['State'].count().sort_values(ascending=False).head(20).index,
    title="Top 20 States with Most Accidents", xlabel='States', ylabel='Count of Accidents')


# In[30]:


bar(x=accident_df.groupby('City')['City'].count().sort_values(ascending=False).head(20).index,
    y=accident_df.groupby('City')['City'].count().sort_values(ascending=False).head(20).values,
   color = accident_df.groupby('City')['City'].count().sort_values(ascending=False).head(20).index,
    title="Top 20 Cities with Most Accidents", xlabel='Cities', ylabel='Count of Accidents')


# In[31]:


bar(x=high_severity_.groupby('City')['City'].count().sort_values(ascending=False).head(20).index,
    y=high_severity_.groupby('City')['City'].count().sort_values(ascending=False).head(20).values,
   color = high_severity_.groupby('City')['City'].count().sort_values(ascending=False).head(20).index,
    title="Top 20 Cities with Most Severity 3 & 4 Accidents", xlabel='Cities', ylabel='Count of Accidents')


# In[32]:


bar(x=accident_df.groupby(['Timezone'])['Timezone'].count().index,
   y=accident_df.groupby(['Timezone'])['Timezone'].count().values,
   color=accident_df.groupby(['Timezone'])['Timezone'].count().index,
   title="Accidents by Timezone", xlabel='Timezone', ylabel='Count of Accidents')


# In[33]:


accident_df.groupby('Weather_Condition')['Weather_Condition'].count().sort_values(ascending=False).head(10)


# In[34]:


bar(x=accident_df.groupby('Weather_Condition')['Weather_Condition'].count().sort_values(ascending=False).head(10).index,
   y=accident_df.groupby('Weather_Condition')['Weather_Condition'].count().sort_values(ascending=False).head(10).values,
   color=accident_df.groupby('Weather_Condition')['Weather_Condition'].count().sort_values(ascending=False).head(10).index,
   title='Top 10 Most Common Weather Conditions When Accidents Occur', xlabel='Weather Condition', ylabel='Count of Accidents')


# In[35]:


fig = make_subplots(rows=2, cols=2, subplot_titles=("Number of Accidents by Year", "Number of Accidents by Month",
                                                    "Number of Accidents by Day of Week", "Number of Accidents by the Hour"))

fig.append_trace(go.Bar(
    x=accident_df.groupby('Year')['Year'].count().index,
    y=accident_df.groupby('Year')['Year'].count().values,
    name='Year'),
    row=1, col=1)

fig.append_trace(go.Bar(
    x=accident_df.groupby('Month')['Month'].count().index,
    y=accident_df.groupby('Month')['Month'].count().values,
    name='Month'),
    row=1, col=2)

fig.append_trace(go.Bar(
    x=accident_df.groupby('DayofWeek',sort=False)['DayofWeek'].count().index,
    y=accident_df.groupby('DayofWeek',sort=False)['DayofWeek'].count().values,
    name='Day of Week'),
    row=2, col=1)

fig.append_trace(go.Bar(
    x=accident_df.groupby('Hour')['Hour'].count().index,
    y=accident_df.groupby('Hour')['Hour'].count().values,
    name='Hour'),
    row=2, col=2)

fig.update_layout(height=1200, width=1200, xaxis_showticklabels=True)
fig.show()


# In[36]:


fig = make_subplots(rows=2, cols=2, subplot_titles=('Sunrise Sunset','Civil Twilight',' Nautical Twilight','Astronomical Twilight'))

fig.append_trace(go.Bar(
    x=accident_df.groupby('Sunrise_Sunset')['Sunrise_Sunset'].count().index,
    y=accident_df.groupby('Sunrise_Sunset')['Sunrise_Sunset'].count().values,
    name='Year'),
    row=1, col=1)

fig.append_trace(go.Bar(
    x=accident_df.groupby('Civil_Twilight')['Civil_Twilight'].count().index,
    y=accident_df.groupby('Civil_Twilight')['Civil_Twilight'].count().values,
    name='Month'),
    row=1, col=2)

fig.append_trace(go.Bar(
    x=accident_df.groupby('Nautical_Twilight')['Nautical_Twilight'].count().index,
    y=accident_df.groupby('Nautical_Twilight')['Nautical_Twilight'].count().values,
    name='Day of Week'),
    row=2, col=1)

fig.append_trace(go.Bar(
    x=accident_df.groupby('Astronomical_Twilight')['Astronomical_Twilight'].count().index,
    y=accident_df.groupby('Astronomical_Twilight')['Astronomical_Twilight'].count().values,
    name='Hour'),
    row=2, col=2)


fig.update_layout(height=1200, width=1200, xaxis_showticklabels=True)
fig.show()


# In[38]:


accident_df.groupby('Severity')[['Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)']].mean()


# In[39]:


del high_severity_
del cols
del fig
gc.collect()


# In[160]:


# accident_df = reduce_mem_usage(accident_df)


# In[40]:


accident_df['Severity'].value_counts()


# In[41]:


#create a small straitifed sample
severity_1 = len(accident_df[accident_df['Severity'] == 1])
severity_2 = len(accident_df[accident_df['Severity'] == 2])
severity_3 = len(accident_df[accident_df['Severity'] == 3])
severity_4 = len(accident_df[accident_df['Severity'] == 4])

accident_df_1 = accident_df[accident_df['Severity']==1].head(round(severity_1/50))
accident_df_2 = accident_df[accident_df['Severity']==2].head(round(severity_2/50))
accident_df_3 = accident_df[accident_df['Severity']==3].head(round(severity_3/50))
accident_df_4 = accident_df[accident_df['Severity']==4].head(round(severity_4/50))

processed_accident_df = pd.concat([accident_df_1, accident_df_2,
                                accident_df_3, accident_df_4], ignore_index=True)

del accident_df_1
del accident_df_2
del accident_df_3
del accident_df_4

gc.collect()


# In[42]:


#create a large straitified sample
large_accident_df_1 = accident_df[accident_df['Severity']==1].head(round(severity_1/20))
large_accident_df_2 = accident_df[accident_df['Severity']==2].head(round(severity_2/20))
large_accident_df_3 = accident_df[accident_df['Severity']==3].head(round(severity_3/20))
large_accident_df_4 = accident_df[accident_df['Severity']==4].head(round(severity_4/20))

large_accident_df = pd.concat([large_accident_df_1, large_accident_df_2,
                                large_accident_df_3, large_accident_df_4], ignore_index=True)
del large_accident_df_1
del large_accident_df_2
del large_accident_df_3
del large_accident_df_4
del severity_1
del severity_2
del severity_3
del severity_4

gc.collect()


# In[165]:


def create_dummy(df, column_name):
    """
    create dummy variables
    
    Args: df: dataframe, col: columns to transform to dummy variables
    
    Returns: dataframe with the dummy variable columns 
    """
    dummies = pd.get_dummies(df[column_name], prefix = column_name)
    df = pd.concat([df, dummies], axis = 1)
    return df


# In[43]:


def train_model(algo_name, model, X_train, y_train):
    """
    Fits the estimator and calculates the average f1 score accross 5-fold cross validations.
    
    Args: algo_name: string name for algorithm, model: estimator,
    X_train: the training part of the first sequence (X),
    y_train: the training part of teh second sequence (y)
    
    Returns: A dataframe with the algoithm's name, average f1 score across 5-fold cross validations.
    """
    model.fit(X_train, y_train)
    avg_f1 = cross_val_score(model, X_train, y_train, scoring = make_scorer(f1_score, average='weighted', labels=[2]), cv = 5).mean()
    
    train_model_results = pd.DataFrame({
        'Algorithm':["{}".format(algo_name)],
        'CV F1':[avg_f1]
    })
    
    return train_model_results, model


# In[107]:


processed_accident_df_X = processed_accident_df.drop('Severity', axis=1)
processed_accident_df_y = processed_accident_df['Severity']


# In[108]:


processed_accident_df_X


# In[109]:


from collections import Counter
from imblearn.under_sampling import RandomUnderSampler


print('Original dataset shape %s' % Counter(processed_accident_df_y))

ros = RandomUnderSampler(random_state=42) # default sampling strategy: resample all classes but the majority class
processed_accident_df_X, processed_accident_df_y = ros.fit_resample(processed_accident_df_X, processed_accident_df_y)

print('Resampled train dataset shape %s' % Counter(processed_accident_df_y))


# In[110]:


processed_accident_df_X


# In[112]:


transformer = LabelBinarizer()
binary_cols = ['Amenity','Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
       'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']

for col in binary_cols:
    processed_accident_df_X[col] = transformer.fit_transform(processed_accident_df_X[col]).ravel()


# In[111]:


processed_accident_df_X


# In[113]:


ohe = OneHotEncoder(sparse = False)

ohe_cols =['City','County','State', 'Timezone', 'Wind_Direction', 'Weather_Condition',
           'Sunrise_Sunset', 'Civil_Twilight','Nautical_Twilight', 'Astronomical_Twilight',
            'Year', 'Month', 'Day', 'Hour', 'DayofWeek']

column_trans = make_column_transformer((
                ohe, ohe_cols), remainder='passthrough')

processed_accident_df_X = column_trans.fit_transform(processed_accident_df_X)
column_names = column_trans.get_feature_names_out()

processed_accident_df_X = pd.DataFrame(processed_accident_df_X, columns=column_names)


# In[80]:


processed_accident_df_X


# In[81]:


scaler = MinMaxScaler()

processed_accident_df_X_sc = scaler.fit_transform(processed_accident_df_X)


# In[82]:


processed_accident_df_X.shape


# In[83]:


untuned_DT, model_DT = train_model(algo_name = 'Untuned Decision Tree', model = DecisionTreeClassifier() , X_train = processed_accident_df_X, y_train = processed_accident_df_y)
untuned_DT


# In[86]:


untuned_KNN, model_KNN = train_model(algo_name = 'Untuned KNN', model = KNeighborsClassifier() , X_train = processed_accident_df_X_sc, y_train = processed_accident_df_y)
untuned_KNN


# In[87]:


untuned_LR, model_LR = train_model(algo_name = 'Untuned Logistic Regression', model = LogisticRegression() , X_train = processed_accident_df_X, y_train = processed_accident_df_y)
untuned_LR


# In[88]:


training_results = pd.concat([
                            untuned_KNN,
                            untuned_DT,
                            untuned_LR], axis=0, ignore_index=True)
training_results


# In[89]:


del untuned_KNN
del untuned_DT
del untuned_LR
gc.collect()


# PARAMETER OPTIMIZATION

# In[90]:


############################### Import Libraries & Modules #################################
from sklearn.tree import DecisionTreeClassifier # A decision tree classifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
# Pipeline of transforms with a final estimator
from sklearn.pipeline import Pipeline
np.random.seed(42) # ensure reproducability

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier # A decision tree classifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

random_state=42


# In[92]:


#Inner and outer 5 cross-validation splits
inner_cv = KFold(n_splits=5,shuffle=True,random_state=random_state)
outer_cv = KFold(n_splits=5,shuffle=True,random_state=random_state)


# In[96]:


import warnings
warnings.filterwarnings('ignore')
#Normalize Data
pipe = Pipeline([
        ('sc', StandardScaler()),
        ('knn', KNeighborsClassifier(p=2,
                                     metric='minkowski'))
      ])
##
params = {
        'knn__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21],
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

gs_knn2 = GridSearchCV(estimator=pipe,
                  param_grid=params,
                  scoring='f1_macro',
                  cv=inner_cv,
                  n_jobs=4)
gs_knn2 = gs_knn2.fit(processed_accident_df_X,processed_accident_df_y)
print("\n Parameter Tuning for KNN")
print("Non-nested CV F1: ", gs_knn2.best_score_)
print("Optimal Parameter: ", gs_knn2.best_params_)
print("Optimal Estimator: ", gs_knn2.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score
nested_score_gs_knn2 = cross_val_score(gs_knn2, X=processed_accident_df_X, y=processed_accident_df_y, cv=outer_cv)
print("Nested CV F1: ",nested_score_gs_knn2.mean(), " +/- ", nested_score_gs_knn2.std())


# In[97]:


gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                               'min_samples_leaf':[1,2,3,4,5]
                               }],
                  scoring='f1_macro',
                  cv=inner_cv)

gs = gs.fit(processed_accident_df_X,processed_accident_df_y)
print("Parameter Tuning for Decision Tree")
print("Non-nested CV F1-score: ", gs.best_score_)
print("Optimal Parameter: ", gs.best_params_)    # Parameter setting that gave the best results on the hold out data.
print("Optimal Estimator: ", gs.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score
nested_score_gs = cross_val_score(gs, X=processed_accident_df_X, y=processed_accident_df_y,scoring='f1_macro',  cv=outer_cv)
print("Nested CV F1-score: ",nested_score_gs.mean(), " +/- ", nested_score_gs.std())


# In[100]:


############################ Logistic Regression Parameter Tuning ############################
#To ignore the convergence warnings
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

gs_lr2 = GridSearchCV(estimator=LogisticRegression(random_state=42, solver='liblinear'),
                  param_grid=[{'C': [  0.00001, 0.0001, 0.001, 0.01, 0.1 ,1 ,10 ,100, 1000, 10000, 100000, 1000000, 10000000],
                              'penalty':['l1','l2','elasticnet','None'],
                               'multi_class':['auto','ovr','multinomial']}],
                  scoring='f1_macro',
                  cv=inner_cv)

gs_lr2 = gs_lr2.fit(processed_accident_df_X,processed_accident_df_y)
print("\n Parameter Tuning Logistic Regression")
print("Non-nested CV F1: ", gs_lr2.best_score_)
print("Optimal Parameter: ", gs_lr2.best_params_)
print("Optimal Estimator: ", gs_lr2.best_estimator_)
nested_score_gs_lr2 = cross_val_score(gs_lr2, X=processed_accident_df_X, y=processed_accident_df_y, cv=outer_cv)
print("Nested CV F1:",nested_score_gs_lr2.mean(), " +/- ", nested_score_gs_lr2.std())


# In[101]:


from tabulate import tabulate
table = [['Classifier', 'mean', '+/- STD'],
         ['k-NN', nested_score_gs_knn2.mean(), nested_score_gs_knn2.std()],
         ['Decision Tree', nested_score_gs.mean(), nested_score_gs.std()],
         ['Logistic Regression', nested_score_gs_lr2.mean(), nested_score_gs_lr2.std()]]
print(tabulate(table, headers="firstrow"))

