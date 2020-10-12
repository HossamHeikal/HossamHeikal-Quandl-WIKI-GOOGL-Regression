import pandas as pd
import math
import quandl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
quandl.ApiConfig.api_key = "Your Quandl API"
df = quandl.get('WIKI/GOOGL')


df['HL_pct'] = ((df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'])*100
df['pct_change'] = ((df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'])*100
df = df[['Adj. Close', 'HL_pct', 'pct_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
label =  df[forecast_col].shift(-forecast_out)
df_1 = pd.merge(df, label, how='inner', on=['Date', 'Date'])

df_1.dropna(inplace = True)

# Making features array except for the label column

x = np.array(df_1.drop(['Adj. Close_y'], 1))
y = np.array(df_1['Adj. Close_y'])

x = preprocessing.scale(x)
scores = cross_validate(LinearRegression(), x, y, cv=5, return_estimator = True)

#for key in scores.keys():
 #  for value in len(scores.values()):
  #     if key == 'test_score':
def get_model(scores):       
    z = scores["test_score"] == scores['test_score'].max()
    index = 0
    for i in range(len(z)-1):
        if z[i] == True:
            index = i
    return scores['estimator'][index]

y_pred = get_model(scores).predict(x)

plt.scatter(y, y_pred)
plt.show()
#return mod_indx
        
#y_pred = cross_val_predict(LinearRegression, x, y, cv=5)