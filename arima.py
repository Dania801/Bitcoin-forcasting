import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn; seaborn.set()
from sklearn.metrics import mean_squared_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

data = pd.read_csv("../data/bitcoin_price_Training - Training.csv")
print(data.head(5))
print(data.tail(5))
data.dtypes
data.info()
data.describe()

data = pd.read_csv("../data/bitcoin_price_Training - Training.csv",index_col= 'Date')
print(data.head(5)) 
data.info()
data.index = pd.to_datetime(data.index)
print(data.index)
data.head(5)

data = data.sort_index()
data.head()

plt.ylabel("DAily Bitcoin price")
plt.plot(data['Close'])
plt.show()

weekly = data.resample('W').sum()
plt.ylabel('Weekly bitcoin price')
plt.plot(weekly)
plt.show()

by_year = data.groupby(data.index.year).mean()
plt.plot(by_year)
plt.show()

by_day = data.groupby(data.index.dayofyear).mean()
plt.plot(by_day)
plt.show()

by_month = data.groupby(data.index.month).mean()
plt.plot(by_month)
plt.show()

def testStationarity(timeSeries):
    rollingMean = timeSeries.rolling(12, center=False).mean()
    rollingStd = timeSeries.rolling(12, center=False).std()
    plt.figure(figsize=(15,6))
    originalPlot = plt.plot(timeSeries, color='blue',label='Original')
    meanPlot = plt.plot(rollingMean, color='red', label='Rolling Mean')
    stdPlot = plt.plot(rollingStd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
def dickeyFullerTest(timeSeries):
    print('Results of Dickey-Fuller Test:')
    fullerTest = adfuller(timeSeries, autolag='AIC')
    fullerOutput = pd.Series(fullerTest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in fullerTest[4].items():
        fullerOutput['Critical Value (%s)'%key] = value
    print(fullerOutput)

testStationarity(data['Close'])
dickeyFullerTest(data['Close'])

ts_logtransformed = np.log(data['Close'])
plt.plot(ts_logtransformed)
plt.show()

ts_logtransformed.head(10)
Rolling_average = ts_logtransformed.rolling(window = 7, center= False).mean()
plt.plot(ts_logtransformed, label = 'Log Transformed')
plt.plot(Rolling_average, color = 'red', label = 'Rolling Average')
plt.legend(loc = 'best')
plt.show()

log_Rolling_difference = ts_logtransformed - Rolling_average
log_Rolling_difference.head(10)
log_Rolling_difference.tail(10)

log_Rolling_difference.dropna(inplace=True)
plt.plot(log_Rolling_difference)

testStationarity(log_Rolling_difference)
dickeyFullerTest(log_Rolling_difference)

expwighted_avg = ts_logtransformed.ewm(halflife=7,min_periods=0,adjust=True,ignore_na=False).mean()
plt.plot(ts_logtransformed, label = 'Log transfomed')
plt.plot(expwighted_avg, color='red', label = 'exponential weighted average')
plt.legend(loc = 'best')
plt.show()

log_expmovwt_diff = ts_logtransformed - expwighted_avg

testStationarity(log_expmovwt_diff)
dickeyFullerTest(log_expmovwt_diff)

ts_diff_logtrans = ts_logtransformed -ts_logtransformed.shift(7)
plt.plot(ts_diff_logtrans)
ts_diff_logtrans.head(10)
ts_diff_logtrans.dropna(inplace=True)
testStationarity(ts_diff_logtrans)
dickeyFullerTest(ts_diff_logtrans)

decomposition = seasonal_decompose(ts_logtransformed)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_logtransformed, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

decomposed_TS = residual
decomposed_TS.dropna(inplace=True)
testStationarity(decomposed_TS)
dickeyFullerTest(decomposed_TS)

#ACF and PACF plots:
lag_acf = acf(ts_diff_logtrans, nlags=30)
lag_pacf = pacf(ts_diff_logtrans, nlags=50, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.show()

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
pyplot.figure()
pyplot.subplot(211)
plot_acf(ts_diff_logtrans, ax=pyplot.gca(),lags=40)
pyplot.subplot(212)
plot_pacf(ts_diff_logtrans, ax=pyplot.gca(), lags=50)
pyplot.show()


model = ARIMA(ts_logtransformed, order=(8,2,1))  
results_ARIMA = model.fit(disp=-1, trend='nc')  
plt.plot(ts_diff_logtrans)
plt.plot(results_ARIMA.fittedvalues, color='red', label = 'order 8')
RSS = results_ARIMA.fittedvalues-ts_diff_logtrans
RSS.dropna(inplace=True)
plt.title('RSS: %.4f'% sum(RSS**2))
plt.legend(loc = 'best')
plt.show()

plt.plot(ts_logtransformed, label = 'log_tranfromed_data')
plt.plot(results_ARIMA.resid, color ='green',label= 'Residuals')
plt.title('ARIMA Model Residual plot')
plt.legend(loc = 'best')
plt.show()

results_ARIMA.resid.plot(kind='kde')
plt.title('Density plot of the residual error values')
print(results_ARIMA.resid.describe())


test = pd.read_csv("../data/bitcoin_price_1week_Test - Test.csv",index_col= 'Date')
test.index = pd.to_datetime(test.index)
test = test['Close']
test = test.sort_index()
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(ts_logtransformed.iloc[0], index=ts_logtransformed.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)

dates = [pd.Timestamp('2017-08-01'), pd.Timestamp('2017-08-02'), pd.Timestamp('2017-08-03'),pd.Timestamp('2017-08-04'), pd.Timestamp('2017-08-05'), pd.Timestamp('2017-08-06'), pd.Timestamp('2017-08-07'), pd.Timestamp('2017-08-30')]

forecast = pd.Series(results_ARIMA.forecast(steps=7)[0],dates)
forecast = np.exp(forecast)
print(forecast)
error = mean_squared_error(test, forecast)
print('Test MSE: %.3f' % error)
