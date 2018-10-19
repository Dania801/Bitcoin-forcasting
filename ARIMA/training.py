import warnings
import pandas
import numpy
import visualize
import stationarize
import matplotlib.pylab as plot
from matplotlib.pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 15,6


def plotACF(rollingData):
    lag = 20
    lagAcf = acf(rollingData, nlags=lag)
    plot.figure(figsize=(15,3))
    plot.plot(lagAcf)
    plot.axhline(y=0,linestyle='--',color='gray')
    plot.axhline(y=-1.96/numpy.sqrt(len(rollingData)),linestyle='--',color='gray')
    plot.axhline(y=1.96/numpy.sqrt(len(rollingData)),linestyle='--',color='gray')
    plot.title('ACF')
    plot.tight_layout()
    plot.show()

    plot.figure(figsize=(15,3))
    plot_acf(rollingData, ax=plot.gca(),lags=lag)
    plot.show()

def plotPACF(rollingData):
    lag = 20
    lagPacf = pacf(rollingData, nlags=lag, method='ols')
    plot.figure(figsize=(15,3))
    plot.plot(lagPacf)
    plot.axhline(y=0,linestyle='--',color='gray')
    plot.axhline(y=-1.96/numpy.sqrt(len(rollingData)),linestyle='--',color='gray')
    plot.axhline(y=1.96/numpy.sqrt(len(rollingData)),linestyle='--',color='gray')
    plot.title('PACF')
    plot.tight_layout()
    plot.show()
    plot.figure(figsize=(15,3))
    plot_pacf(rollingData, ax=plot.gca(), lags=lag)
    plot.tight_layout()
    plot.show()

def ARModel(rollingData):
    p=2
    q=2
    d=1
    model = ARIMA(rollingData, order=(p, d, 0))  
    resultAR = model.fit(disp=-1)
    plot.figure(figsize=(15,6))
    plot.plot(rollingData)
    plot.plot(resultAR.fittedvalues, color='red')
    plot.title('RSS: %.4f'% sum((resultAR.fittedvalues-rollingData.Weighted_Price).dropna()**2))
    plot.show()
    print (resultAR.summary())

def MAModel(rollingData):
    p=2
    q=2
    d=1
    model = ARIMA(rollingData, order=(0, d, q))  
    resultMA = model.fit(disp=-1) 
    plot.figure(figsize=(15,6))
    plot.plot(rollingData)
    plot.plot(resultMA.fittedvalues, color='red')
    plot.title('RSS: %.4f'% sum((resultMA.fittedvalues-rollingData.Weighted_Price).dropna()**2))
    plot.show()
    print (resultMA.summary())

def ARIMAModel(rollingData):
    p=2
    q=2
    d=1
    model = ARIMA(rollingData, order=(p, d, q))  
    resultARIMA = model.fit(disp=-1, trend='nc')
    plot.figure(figsize=(15,6))
    plot.plot(rollingData, label='rollingData_rolling')
    plot.plot(resultARIMA.fittedvalues, color='red')
    plot.title('RSS: %.4f'% sum((resultARIMA.fittedvalues-rollingData.Weighted_Price).dropna()**2))
    plot.legend(loc='best')
    plot.show()
    return resultARIMA

def plotResiduals(modelResult, rollingData):
    data = pandas.DataFrame(modelResult.fittedvalues)
    data.columns = rollingData.columns
    data = data - rollingData
    # data = data.cumsum()
    plot.plot(data, label='residuals')
    plot.legend(loc='best')
    plot.show()

def trainingScript():
    rollingData = stationarize.stationarizeData()
    plotACF(rollingData)
    plotPACF(rollingData)
    ARModel(rollingData)
    model = ARIMAModel(rollingData)
    plotResiduals(model, rollingData)

def getModel():
    rollingData = stationarize.stationarizeData()
    model = ARIMAModel(rollingData)
    return model