# Regression using cross_validate on WIKI/GOOGL Quandl Data

# Importing essential libraries
import pandas as pd
import math
import quandl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

# Connect to your Quandl account with API(Not mandatory for les that 50 requests a day)
quandl.ApiConfig.api_key = "Your Quandl API"
df = quandl.get('WIKI/GOOGL')

# Calculations
df['HL_pct'] = ((df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'])*100
df['pct_change'] = ((df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'])*100
df = df[['Adj. Close', 'HL_pct', 'pct_change', 'Adj. Volume']]

# Identifying forecast column
forecast_col = 'Adj. Close'
# make NA values very small and not remove it as it may effect the analysis if removed
df.fillna(-99999, inplace=True)

# How many days ahead are being predected
forecast_out = int(math.ceil(0.01*len(df)))

# Shift columns and append it to the dataframe
label =  df[forecast_col].shift(-forecast_out)
df_1 = pd.merge(df, label, how='inner', on=['Date', 'Date'])
df_1.dropna(inplace = True)

# Making features array except for the label column

x = np.array(df_1.drop(['Adj. Close_y'], 1))
y = np.array(df_1['Adj. Close_y'])

# Preprocessing to speed up analysis
x = preprocessing.scale(x)
scores = cross_validate(LinearRegression(), x, y, cv=5, return_estimator = True)

# Get model function
def get_model(scores):       
    z = scores["test_score"] == scores['test_score'].max()
    index = 0
    for i in range(len(z)-1):
        if z[i] == True:
            index = i
    return scores['estimator'][index]

# Predictinng x with the highest score model
y_pred = get_model(scores).predict(x)

# Making a Dataframe
A = pd.DataFrame(y, columns = ['y'])
B = pd.DataFrame(y_pred, columns = ['y_pred'])
C = pd.concat([A, B], axis = 1, join = 'inner')

# Plot
sns.scatterplot(data = C, x = "y" , y = "y_pred", hue = 'y')
sns.lineplot(x = np.array([0,1200]), y = np.array([0, 1200]))