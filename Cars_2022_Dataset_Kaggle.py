#!/usr/bin/env python
# coding: utf-8

# # Cars 2022 Dataset Kaggle

# Link for the dataset:
# https://www.kaggle.com/datasets/tr1gg3rtrash/cars-2022-dataset

# ## About Dataset

# This dataset was derived from https://www.cardekho.com/. The dataset consists specifications list as we all rating only for new cars.
# 
# I would like to thank https://www.cardekho.com/ for such a great platform. Looking forward to many notebooks on the dataset in near future.

# ## Dataset Description

# This dataset consists of 16 features on a total of 203 car choices available in India.

# | **Feature** | **Type**   | **Description** |
# | :-----------| :--------: | :---------------------- |
# | car_name | String | Name of the car |
# | reviews_count| String | Number of reviews given to the specific car on the website |
# | fuel_type| String | Type of Fuel car uses. Possible values are Petrol, Diesel and Electric |
# | engine_displacement | Integer | Engine displacement is the measure of the cylinder volume swept by all of the pistons of a piston engine, excluding the combustion chambers. Unit is (cc) |
# | no_cylinder |Integer  | Number of cylinders contained by the car. 0 in case of electric vehicles |
# | seating_capacity | Integer | Number of people that can fit in the car |
# | transmission_type |String  | Possible values range from Manual, Automatic and Electric |
# | fuel_tank_capacity | Integer | Maximum capacity of car's fuel tank. 0 in case of electric vehicle |
# | body_type | String | Body shape of the car |
# | rating | Integer |  Rating provided to the car on the website. In the range of 0 to 5|
# | starting_price | Integer | Starting price of the car in Rs |
# | ending_price | Integer | Ending price of the car in Rs |
# | max_torque_nm | Integer | Maximum torque that can be provided by the car |
# | max_torque_rpm | Integer | RPM at which maximum torque can be achieved |
# | max_power_bhp | Integer | Maximum horsepower of the car |
# | max_power_rp | Integer | RPM at which maximum horsepower can be achieved |

# ## Data and Setup

# Import numpy and pandas
import pandas as pd
import numpy as np


# Import visualization libraries and set %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# Set the parameters that control the general style of the plots.
# sns.set_style('whitegrid')
# The %matplotlib inline command tells the IPython environment to draw the plots immediately after the current cell. 
# get_ipython().run_line_magic('matplotlib', 'inline')

# Read Data from 'CARS_1.csv' file as a DataFrame called df
df = pd.read_csv('CARS_1.csv')

# Check the top five datafrom df
df.head(5)

# Check the info() of the 'df'
df.info()

# Calculate some statistical data like percentile, mean, and standard deviation of the numerical values of the Series or DataFrame
df.describe()

# Transpose of describe()
df.describe().transpose()

# ### Average Price Calculation

# Average starting price
df['starting_price'].mean()

# Minimum and Maximum of starting price
df['starting_price'].min()

df['starting_price'].max()

# Average ending price
df['ending_price'].mean()

# Minimum and Maximum of ending price
df['ending_price'].min()

df['ending_price'].max()

df[df['car_name']=='Maruti'].count()

# ## Basic Questions

# What are the car names?
df['car_name'].value_counts()


# What are the top 5 car name?
df['car_name'].value_counts().head(5)

# **What are the FUEL type?**
df['fuel_type'].value_counts()

df['fuel_type'].nunique()

# What are the top 5 most popular cars
df['car_name'].apply(lambda x:x.split(' ')[0])

df['car_name'].apply(lambda x:x.split(' ')[0]).value_counts()

df['car_name'].apply(lambda x:x.split(' ')[0]).value_counts().head(5)

# **What are the transmission_type?**
df['transmission_type'].value_counts()

# **What are the body_type?**
df['body_type'].value_counts()

# ## Plot Types
# There are several plot types built-in to pandas, most of them statistical plots by nature:
# 
# - df.plot.area
# - df.plot.barh
# - df.plot.density
# - df.plot.hist
# - df.plot.line
# - df.plot.scatter
# - df.plot.bar
# - df.plot.box
# - df.plot.hexbin
# - df.plot.kde
# - df.plot.pie
# 
# You can also just call df.plot(kind='hist') or replace that kind argument with any of the key terms shown in the list above (e.g. 'box','barh', etc..)
df['rating'].plot.hist() # bins=50

df.plot.scatter(x='seating_capacity', y='ending_price')

# ## Heatmap

# Matrix form for correlation data
df.corr()

# Heatmap for correlation data
sns.heatmap(df.corr())

fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, ax=ax)

# pairplot
sns.pairplot(data=df)

sns.jointplot(x='starting_price', y='engine_displacement', data=df)

sns.barplot(x='transmission_type',y='starting_price', data=df)

sns.barplot(x='transmission_type',y='ending_price', data=df)

sns.barplot(x='fuel_type', y='max_torque_nm', data=df)

fig, ax = plt.subplots(figsize=(10,7))
sns.barplot(x='body_type', y='max_torque_nm', data=df)

sns.boxplot(x='transmission_type', y='ending_price',data=df, palette='rainbow')

# Can do entire dataframe with orient='h'
sns.boxplot(data=df,palette='rainbow',orient='h')

# ### Categorical Data Plots
# Now let's discuss using seaborn to plot categorical data! There are a few main plot types for this:
# 
# - factorplot
# - boxplot
# - violinplot
# - stripplot
# - swarmplot
# - barplot
# - countplot

# ### violinplot
# A violin plot plays a similar role as a box and whisker plot. It shows the distribution of quantitative data across several levels of one (or more) categorical variables such that those distributions can be compared. Unlike a box plot, in which all of the plot components correspond to actual datapoints, the violin plot features a kernel density estimation of the underlying distribution.

sns.violinplot(x="transmission_type", y="starting_price", data=df,palette='rainbow')

sns.violinplot(x='transmission_type', y='ending_price', data=df, hue='fuel_type', palette='Set1')

# ### stripplot and swarmplot
# The stripplot will draw a scatterplot where one variable is categorical. A strip plot can be drawn on its own, but it is also a good complement to a box or violin plot in cases where you want to show all observations along with some representation of the underlying distribution.
# 
# The swarmplot is similar to stripplot(), but the points are adjusted (only along the categorical axis) so that they don’t overlap. This gives a better representation of the distribution of values, although it does not scale as well to large numbers of observations (both in terms of the ability to show all the points and in terms of the computation needed to arrange them).

sns.stripplot(x='fuel_type', y='max_power_bhp', data=df)

sns.stripplot(x='transmission_type', y='starting_price', data=df)

fig, ax = plt.subplots(figsize=(8,8))
sns.swarmplot(x='transmission_type', y='ending_price', hue='fuel_type', data=df, palette='Set1', split=True)

# ### Combining Categorical Plots
sns.violinplot(x='max_power_bhp', y='body_type', data=df, palette='rainbow')
sns.swarmplot(x='max_power_bhp', y='body_type', data=df, color='black', size=3)

# ### factorplot
# factorplot is the most general form of a categorical plot. It can take in a kind parameter to adjust the plot type:
sns.factorplot(x='transmission_type', y='starting_price', data=df, kind='bar')

# ### FacetGried
g = sns.FacetGrid(data=df,col='transmission_type')
g.map(plt.hist,'starting_price')

sns.distplot(df['starting_price'])

# ## Data Pre-Processing
df.info()

df['car_name'].value_counts()

# ### Converting Categorical values into numerical value

# Convert Fuel Types into Numerical Features
df['fuel_type'].value_counts()

df['fuel_type'].replace(['Petrol','Diesel','Electric','CNG'],[0,1,2,3], inplace=True)

# Convert Transmission Types into numerical feature

df['transmission_type'].value_counts()

df['transmission_type'].replace(['Automatic','Manual','Electric'],[0,1,2], inplace=True)

# Convert Body type into numerical feature
df['body_type'].value_counts()

df['body_type'].replace(['SUV','Sedan','Hatchback','Coupe','MUV','Convertible','Pickup Truck','Luxury','Wagon',
                        'Hybrid','Minivan'],[0,1,2,3,4,5,6,7,8,9,10], inplace=True)


# Checking the data types of feature after pre-processing operation
df.info()

df.head(5)

# #### Drop out the car_name

# Drop out the car_name feature because of categorical nature
df.drop('car_name', axis=1, inplace=True)


# ### Check info after all pre-processing operation
df.info()

df.describe().transpose()

# ### Check NaN columns in Features

# Check for NaN under an entire DataFrame
df.isnull().values.any()

# Count the NaN under an entire DataFrame
df.isnull().sum()

df.isnull().sum().sum()

# ### Fill NaN value with mean() value
df['seating_capacity'].fillna(value=df['seating_capacity'].mean(), inplace=True)

# ## Predict the strating and ending price

# ### Training a Linear Regression Model
# 
# Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the starting and ending prices column.

# Split data (df) into X and y arrays

X = df[['reviews_count','fuel_type','engine_displacement','no_cylinder','seating_capacity','transmission_type',
       'fuel_tank_capacity','body_type','rating','max_torque_nm','max_torque_rpm','max_power_bhp','max_power_rp']]
y = df[['starting_price','ending_price']]

# ### Train Test Split
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)

# ### Creating and Traning Linear Regression Model

# ** Import LinearRegression from sklearn.linear_model **
from sklearn.linear_model import LinearRegression

# Create an instance of a LinearRegression() model named lm.
lm = LinearRegression()

# ** Train/fit lm on the training data.**
lm.fit(X_train,y_train)

# **Print out the coefficients of the model**
print('Coefficients: \n', lm.coef_)

# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

#We can't use the following code because of two output
#coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
# coeff_df

# ValueError: Shape of passed values is (2, 13), indices imply (13, 1)

# ### Predicting the Test Data 
# 
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**
predictions = lm.predict(X_test)

# * Create a scatterplot of the real test values versus the predicted values. **
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

# ### Residuals
# 
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data.
# 
# Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
sns.distplot((y_test-predictions),bins=30)


#
#  
# Comparing these metrics:
# 
# MAE is the easiest to understand, because it's the average error.
# MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
# All of these are loss functions, because we want to minimize them.

# ## Evaluating the Model¶
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# **Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.**

# CALCULATE METRICS
from sklearn import metrics

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, predictions))
print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(y_test, predictions))
print('Mean Square Error (MSE):', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Suared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## Random Search for Prediction

# # Random search model for prediction

# import all the necessary libraries
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import loguniform
from sklearn.model_selection import RepeatedKFold

# Define Model
model = Ridge()

# Divide data into stratified cross validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Define search space for optimizing hyperparameters
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 100)
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]

# define Random Search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, random_state=1)

# Evaluate the performance of the Model
result = search.fit(X, y)

# summarize result
print('Best Score (Negative Mean Absolute Error): %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# ## Grid Search for Prediction

# grid search linear regression model
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

# define model
model = Ridge()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space for optimizing hyperparameters
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]

# define grid search algorithm
search = GridSearchCV(model, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)

# execute search
result = search.fit(X, y)

# summarize result
print('Best Score (Negative Mean Absolute Error): %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
