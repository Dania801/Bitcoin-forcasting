import warnings
import pandas
import numpy
import matplotlib.pylab as plot
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15,6
warnings.filterwarnings('ignore')

def loadData(fileName):
    data = pandas.read_csv('{0}'.format(fileName))
    data['Seconds'] = data.Timestamp.values.astype(int)
    data.Timestamp = pandas.to_datetime(data.Timestamp, unit='s')
    data = data[data.Open.notnull()]
    return data

def setIndex(data):
    data = data.reset_index().drop('index', axis=1).reset_index()
    data['counter'] = data.index
    data = data.drop('index', axis=1)
    data = data.set_index('Seconds')
    originalData = data.copy()
    data = data.reset_index().set_index('Timestamp').resample('D').mean()
    data = pandas.DataFrame(data)
    return (data,originalData)

def visualize_data(data):
    print('entries missing in data: ', sum(data.Weighted_Price.isnull()))
    print ('Number of entries: ', len(data))
    print (data.head())
    plot.subplot(331)
    plot.title('Open')
    plot.plot(data.index, data.Open)
    plot.subplot(332)
    plot.title('High')
    plot.plot(data.index, data.High)
    plot.subplot(333)
    plot.title('Low')
    plot.plot(data.index, data.Low)
    plot.subplot(334)
    plot.title('Close')
    plot.plot(data.index, data.Close)
    plot.subplot(335)
    plot.title('Weighted price')
    plot.plot(data.index, data.Weighted_Price)
    plot.subplot(336)
    plot.title('Volume_(currency)')
    plot.plot(data.index, data['Volume_(Currency)'])
    plot.subplot(337)
    plot.title('Volume_(BTC)')
    plot.plot(data.index, data['Volume_(BTC)'])
    plot.show()

def addDateColumn(data):
    date = data.reset_index().Timestamp.map(lambda y: pandas.to_datetime(y).date())
    date = numpy.asarray(date, dtype=date)
    data['Date'] = date
    return data

def priceDataframe(data):
    data = data.Weighted_Price
    data = pandas.DataFrame(data)
    return data 

def limitPeriod(data, days):
    data = data[:days]
    return data

def plotFigure(data):
    plot.figure(figsize=(15,6))
    plot.plot(data)
    plot.show()

def plotTwoFigues(data1, data2):
    plot.figure(figsize=(15,6))
    plot.plot(data1, color='blue', label='Original')
    plot.plot(data2, color='red', label='New')
    plot.title('Original (blue) vs New (red)')
    plot.legend(loc='best')
    plot.show()

def visualizeScript():
    data = loadData('../../data/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
    print (len(data))
    data,_ = setIndex(data)
    data = addDateColumn(data)
    data = priceDataframe(data)
    data = limitPeriod(data, 500)
    plotFigure(data)
        
def prepareData():
    data = loadData('../../data/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
    data,_ = setIndex(data)
    data = addDateColumn(data)
    data = priceDataframe(data)
    data = limitPeriod(data, 500)
    return data

def prepareTestData():
    data = loadData('../../data/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
    data,_ = setIndex(data)
    data = addDateColumn(data)
    data = priceDataframe(data)
    # data = limitPeriod(data, 500)
    return data.iloc[:int(len(data)/2)]

visualizeScript()