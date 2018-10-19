import numpy
import pickle
import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.layers import Dense 
from keras.layers import LSTM
import matplotlib.pylab as plot


def scaleClosingPrice(dataset):
    closingPrices = dataset.iloc[:, 3:4]
    scaler = MinMaxScaler()
    scaledClosingPrices = scaler.fit_transform(closingPrices)
    return scaledClosingPrices

def scaleOpenPrice(dataset): 
    openPrice = dataset.iloc[:, 1:2]
    scaler = MinMaxScaler()
    scaledOpenPrices = scaler.fit_transform(openPrice)
    return scaledOpenPrices

def divideTrainingSet(dataset):
    trainingSet = dataset[:10000]
    xTrain = trainingSet[0: len(trainingSet) -1]
    yTrain = trainingSet[1: len(trainingSet)]
    xTrain = numpy.reshape(xTrain, (len(xTrain), 1, xTrain.shape[1]))
    return xTrain, yTrain;

def divideTestSet(dataset):
    testSet = dataset[10000:]
    xTest = testSet[0: len(testSet) -1]
    yTest = testSet[1: len(testSet)]
    xTest = numpy.reshape(xTest, (len(xTest), 1, xTest.shape[1]))
    return xTest, yTest

def trainModel(xTrainData, yTrainData):
    model = Sequential()
    model.add(LSTM(256, 
        return_sequences=True, 
        input_shape=(xTrainData.shape[1], xTrainData.shape[2])))
    model.add(LSTM(256))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(xTrainData, 
        yTrainData, 
        epochs=50, 
        batch_size=16, 
        shuffle=False)
    modelJson = model.to_json()
    with open("model_open.json", "w") as jsonFile:
        jsonFile.write(modelJson)
    model.save_weights("model_open.h5")
    print("Model is saved to disk ...")

def readTrainingData():
    jsonFile = open('model_open.json', 'r')
    loadedModelJson = jsonFile.read()
    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    loadedModel.load_weights("model.h5")
    print("The model is loaded from disk ...")
    return loadedModel

def performeTrainingModel():
    data = preprocessing.prepareClient()
    data = preprocessing.setDateAsIndex(data)
    data = preprocessing.parseData(data)
    data = scaleOpenPrice(data)
    xTrain, yTrain = divideTrainingSet(data)
    trainModel(xTrain, yTrain)
