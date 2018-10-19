import keys 
import pandas
from binance.client import Client 
import matplotlib.pylab as plot

def prepareClient():
    binanceClient = Client(keys.APIKey, keys.SecretKey)
    BTCData = binanceClient.get_historical_klines(symbol = 'BTCUSDT', 
        interval = Client.KLINE_INTERVAL_30MINUTE, 
        start_str = '1 year ago UTC')
    return BTCData

def setDateAsIndex(data):
    BTCFrames = pandas.DataFrame(data, columns = ['Open time', 'Open', 'High', 
        'Low', 'Close', 'Volume', 
        'Close time', 'Quote asset volume', 
        'Number of trades', 'Taker buy base asset volume', 
        'Taker buy quote asset volume', 'Ignore'])
    BTCFrames['Open time'] = pandas.to_datetime(BTCFrames['Open time'], unit = 'ms')
    BTCFrames.set_index('Open time', inplace = True)
    return BTCFrames

def visualizeData(dataFrames):
    plot.subplot(331)
    plot.title('Open')
    plot.plot(dataFrames['Open'])
    plot.subplot(332)
    plot.title('High')
    plot.plot(dataFrames['High'])
    plot.subplot(333)
    plot.title('Low')
    plot.plot(dataFrames['Low'])
    plot.subplot(334)
    plot.title('Close')
    plot.plot(dataFrames['Close'])
    plot.subplot(335)
    plot.title('Volume')
    plot.plot(dataFrames['Volume'])
    plot.subplot(336)
    plot.title('Close time')
    plot.plot(dataFrames['Close time'])
    plot.subplot(337)
    plot.title('Quote asset volume')
    plot.plot(dataFrames['Quote asset volume'])
    plot.subplot(338)
    plot.title('Number of trades')
    plot.plot(dataFrames['Number of trades'])
    plot.subplot(339)
    plot.title('Taker buy base asset volume')
    plot.plot(dataFrames['Taker buy base asset volume'])
    plot.show()

def parseData(dataFrames):
    dataFrames['Open'] = dataFrames['Open'].astype(float)
    dataFrames['High'] = dataFrames['High'].astype(float)
    dataFrames['Low'] = dataFrames['Low'].astype(float)
    dataFrames['Close'] = dataFrames['Close'].astype(float)
    dataFrames['Volume'] = dataFrames['Volume'].astype(float)
    dataFrames['Close time'] = dataFrames['Close time'].astype(float)
    dataFrames['Quote asset volume'] = dataFrames['Quote asset volume'].astype(float)
    dataFrames['Number of trades'] = dataFrames['Number of trades'].astype(float)
    dataFrames['Taker buy base asset volume'] = dataFrames['Taker buy base asset volume'].astype(float)
    dataFrames['Taker buy quote asset volume'] = dataFrames['Taker buy quote asset volume']
    dataFrames['Ignore'] = dataFrames['Ignore'].astype(float)
    return dataFrames

# data = prepareClient()
# data = setDateAsIndex(data)
# data = parseData(data)
# visualizeData(data);