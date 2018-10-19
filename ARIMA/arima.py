import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from datetime import date

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import gc

import warnings
warnings.filterwarnings('ignore')

def load_data(filename):
    df = pd.read_csv('{0}'.format(filename))
    df['Seconds'] = df.Timestamp.values.astype(int) # rename field
    df.Timestamp = pd.to_datetime(df.Timestamp, unit='s') # true timestamp
    df = df[df.Open.notnull()] # remove fields w/o data to lower memory requirements

    df = df.reset_index().drop('index', axis=1).reset_index()
    df['counter'] = df.index
    df = df.drop('index', axis=1)
    df = df.set_index('Seconds')
    df_original = df.copy()
    
    # downsample to days
    df = df.reset_index().set_index('Timestamp').resample('D').mean()
    df = pd.DataFrame(df)
    return (df,df_original)

df2,_ = load_data('../../data/coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv')
df4,_ = load_data('../../data/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')

def visualize_data(df4, df2):
    print('entries missing in df2', sum(df2.Weighted_Price.isnull()))
    print('entries missing in df4', sum(df4.Weighted_Price.isnull()))
    print (len(df4), len(df2))
    print (df4.head())
    print (df2.head())

def add_date(df4):
    _ = df4.reset_index().Timestamp.map(lambda y: pd.to_datetime(y).date())
    _ = np.asarray(_, dtype=date)
    df4['Date'] = _
    return df4

df = add_date(df4) 

def create_dataframe(df):
    df = df4.Weighted_Price
    df = pd.DataFrame(df)
    plt.figure(figsize=(15,6))
    plt.plot(df)
    plt.show()
    return df 

df = create_dataframe(df)

def specify_period(df):
    df = df[:380] # take the first xxx days
    plt.figure(figsize=(15,6))
    plt.plot(df)
    plt.show()
    return df

df = specify_period(df)

# source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/ 
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12, center=False).mean()
    rolstd = timeseries.rolling(12, center=False).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,6))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    Dickey_Fuller_test(timeseries)
    
def Dickey_Fuller_test(timeseries):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(df.Weighted_Price)


def log_series(df):
    ts_df_log = np.log(df)
    #test_stationarity(ts_df_log.Weighted_Price)
    plt.figure(figsize=(15,6))
    plt.plot(df, color='blue', label='original')
    plt.plot(ts_df_log, color='red', label='log')
    plt.title('original (blue) vs log (red)')
    plt.legend(loc='best')
    plt.show()
    return ts_df_log

ts_df_log = log_series(df)
Dickey_Fuller_test(ts_df_log.Weighted_Price)

def rolling_avg_normal(ts_df_log):
    window = 7
    Rolling_average = ts_df_log.rolling(window = window, center= False).mean()
    ts_df_log_rolling = Rolling_average.dropna()
    plt.figure(figsize=(15,6))
    plt.plot(ts_df_log, label = 'Log Transformed')
    plt.plot(ts_df_log_rolling, color = 'red', label = 'Rolling Average')
    plt.legend(loc = 'best')
    plt.show()
    return ts_df_log_rolling

ts_df_log_rolling = rolling_avg_normal(ts_df_log)

def rolling_avg_enhanced(ts_df_log):
    window = 7
    shift_by_days = -2
    Rolling_average = ts_df_log.rolling(window = window, center= False).mean()
    ts_df_log_rolling_temp = Rolling_average.shift(shift_by_days).dropna()
    plt.figure(figsize=(15,6))
    plt.plot(ts_df_log, label = 'Log Transformed')
    plt.plot(ts_df_log_rolling_temp, color = 'red', label = 'Rolling Average')
    plt.legend(loc = 'best')
    plt.show()
    return ts_df_log_rolling_temp

ts_df_log_rolling_temp = rolling_avg_enhanced(ts_df_log)

def rolling_avg_diff(ts_df_log, ts_df_log_rolling_temp):
    ts_df_log_rolling = (ts_df_log - ts_df_log_rolling_temp).dropna()
    plt.figure(figsize=(15,6))
    plt.plot(ts_df_log, label = 'Log Transformed')
    plt.plot(ts_df_log_rolling, color = 'red', label = 'Log and Rolling Average Transformed')
    plt.legend(loc = 'best')
    plt.show()
    Dickey_Fuller_test(ts_df_log_rolling.Weighted_Price)
    return ts_df_log_rolling

ts_df_log_rolling = rolling_avg_diff(ts_df_log, ts_df_log_rolling_temp)








lag = 20
lag_pacf = pacf(ts_df_log_rolling, nlags=lag, method='ols')
lag_acf = acf(ts_df_log_rolling, nlags=lag)

lag = 20
lag_pacf = pacf(ts_df_log_rolling, nlags=lag, method='ols')
lag_acf = acf(ts_df_log_rolling, nlags=lag)

#Plot ACF: 
plt.figure(figsize=(15,3))
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_df_log_rolling)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_df_log_rolling)),linestyle='--',color='gray')
plt.title('ACF')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,3))
plot_acf(ts_df_log_rolling, ax=plt.gca(),lags=lag)
plt.show()

#Plot PACF:
plt.figure(figsize=(15,3))
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_df_log_rolling)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_df_log_rolling)),linestyle='--',color='gray')
plt.title('PACF')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,3))
plot_pacf(ts_df_log_rolling, ax=plt.gca(), lags=lag)
plt.tight_layout()
plt.show()

p=2
q=2
d=1

# AR
model = ARIMA(ts_df_log_rolling, order=(p, d, 0))  
results_AR = model.fit(disp=-1)
plt.figure(figsize=(15,6))
plt.plot(ts_df_log_rolling)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_df_log_rolling.Weighted_Price).dropna()**2))
plt.show()
print (results_AR.summary())

model = ARIMA(ts_df_log_rolling, order=(0, d, q))  
results_MA = model.fit(disp=-1) 
plt.figure(figsize=(15,6))
plt.plot(ts_df_log_rolling)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_df_log_rolling.Weighted_Price).dropna()**2))
plt.show()
print (results_MA.summary())

# ARIMA
model = ARIMA(ts_df_log, order=(p, d, q))  
results_ARIMA = model.fit(disp=-1, trend='nc')
plt.figure(figsize=(15,6))
plt.plot(ts_df_log_rolling, label='ts_df_log_rolling')
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_df_log_rolling.Weighted_Price).dropna()**2))
plt.legend(loc='best')
plt.show()

x = pd.DataFrame(results_ARIMA.fittedvalues)
x.columns = ts_df_log_rolling.columns
x = x - ts_df_log_rolling
# x = x.cumsum()
plt.plot(x, label='residuals')
plt.legend(loc='best')
plt.show()

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_df_log.Weighted_Price, index=ts_df_log.index)
predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.figure(figsize=(15,3))
plt.plot(df, label='first 380 days')
plt.plot(predictions_ARIMA, 'r+', label='predicted')
plt.legend(loc='best')
plt.show()

# start = 360
# end = 400
# forecast = results_ARIMA.predict(start=start, end=end)
# f = (forecast + forecast.shift(-1))
# f = f.shift(-3).dropna()
# forecast = f

# plt.figure(figsize=(15,3))
# plt.plot(df[:end].Weighted_Price, label='original data')
# plt.show()
# plt.plot(forecast, color='red', label='predicted')
# plt.legend(loc='best')
# plt.show()
# plt.plot(df4[start:end].Weighted_Price, label='actual')
# plt.legend(loc='best')
# plt.show()