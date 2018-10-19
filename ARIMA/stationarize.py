import visualize
import warnings
import pandas
import numpy
import matplotlib.pylab as plot
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 15,6

def testStationarity(timeSeries):
    rollingMean = timeSeries.rolling(12, center=False).mean()
    rollingStd = timeSeries.rolling(12, center=False).std()
    plot.figure(figsize=(15,6))
    originalPlot = plot.plot(timeSeries, color='blue',label='Original')
    meanPlot = plot.plot(rollingMean, color='red', label='Rolling Mean')
    stdPlot = plot.plot(rollingStd, color='black', label = 'Rolling Std')
    plot.legend(loc='best')
    plot.title('Rolling Mean & Standard Deviation')
    plot.show()
    
def dickeyFullerTest(timeSeries):
    print('Results of Dickey-Fuller Test:')
    fullerTest = adfuller(timeSeries, autolag='AIC')
    fullerOutput = pandas.Series(fullerTest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in fullerTest[4].items():
        fullerOutput['Critical Value (%s)'%key] = value
    print(fullerOutput)

def logSeries(data):
    dataLog = numpy.log(data)
    return dataLog

def normalRollingAvg(logData):
    window = 7
    rollingAvg = logData.rolling(window = window, center= False).mean()
    rollingLogData = rollingAvg.dropna()
    return rollingLogData

def enhancedRollingAvg(logData):
    window = 7
    daysShift = -2
    rollingAvg = logData.rolling(window = window, center= False).mean()
    rollingLogData = rollingAvg.shift(daysShift).dropna()
    return rollingLogData

def rollingAvgDiff(logData, rollingLogData):
    rollingData = (logData - rollingLogData).dropna()
    return rollingData

def stationaizeScript():
    data = visualize.prepareData()
    testStationarity(data.Weighted_Price)
    dickeyFullerTest(data.Weighted_Price)
    logData = logSeries(data)
    visualize.plotTwoFigues(data, logData)
    dickeyFullerTest(data.Weighted_Price)
    rollData = normalRollingAvg(logData)
    visualize.plotTwoFigues(logData, rollData)
    rollDataTemp = enhancedRollingAvg(logData)
    visualize.plotTwoFigues(logData, rollDataTemp)
    rollDiff = rollingAvgDiff(logData, rollDataTemp)
    visualize.plotTwoFigues(logData, rollDiff)
    dickeyFullerTest(rollDiff.Weighted_Price)

def stationarizeData():
    data = visualize.prepareData()
    logData = logSeries(data)
    rollDataTemp = enhancedRollingAvg(logData)
    rollDiff = rollingAvgDiff(logData, rollDataTemp)
    return rollDiff
