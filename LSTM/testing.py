import training
import preprocessing
import matplotlib.pylab as plot


def testModel_close():
    model = training.readTrainingData()
    data = preprocessing.prepareClient()
    data = preprocessing.setDateAsIndex(data)
    data = preprocessing.parseData(data)
    data = training.scaleClosingPrice(data)
    xTest, yTest = training.divideTestSet(data)
    predictedPrices = model.predict(xTest)
    plot.plot(predictedPrices, color='red', label='Predicted price')
    plot.plot(yTest, color='blue', label='Real price')
    plot.title('Predicted vs. real prices')
    plot.legend(loc='best')
    plot.show()

def testModel_open():
    model = training.readTrainingData()
    data = preprocessing.prepareClient()
    data = preprocessing.setDateAsIndex(data)
    data = preprocessing.parseData(data)
    data = training.scaleOpenPrice(data)
    xTest, yTest = training.divideTestSet(data)
    predictedPrices = model.predict(xTest)
    plot.plot(predictedPrices, color='red', label='Predicted price')
    plot.plot(yTest, color='blue', label='Real price')
    plot.title('Predicted vs. real prices')
    plot.legend(loc='best')
    plot.show()
    
testModel_close()