import visualize
import warnings
import pandas
import numpy
import stationarize
import training 
import matplotlib.pylab as plot
from matplotlib.pylab import rcParams

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 15,6

def testData(resultARIMA, data, logData):
    diffARIMA = pandas.Series(resultARIMA.fittedvalues, copy=True)
    diffARIMA_cumsum = diffARIMA.cumsum()
    logARIMA = pandas.Series(logData.Weighted_Price, index=logData.index)
    predictionARIMA = numpy.exp(logARIMA)

    plot.plot(predictionARIMA, label='predicted')
    plot.plot(data, label='first 380 days')
    plot.legend(loc='best')
    plot.show()

    start = 360
    end = 400
    forecast = resultARIMA.predict(start=start, end=end)
    f = (forecast + forecast.shift(-1))
    f = f.shift(-3).dropna()
    forecast = f

    plot.figure(figsize=(15,3))
    plot.plot(data[:end].Weighted_Price, label='original data')
    plot.show()
    plot.plot(forecast, color='red', label='predicted')
    plot.plot(data[start:end].Weighted_Price, label='actual')
    plot.legend(loc='best')
    plot.show()

def testScript():
    results_ARIMA = training.getModel()
    data = visualize.prepareTestData()
    logData = stationarize.logSeries(data)
    testData(results_ARIMA, data, logData)

testScript()