from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential, model_from_json
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd



def loadData(fileName, sequenceLength):
    rawData = pd.read_csv(fileName, dtype = float).values
    
    for row in range(0, rawData.shape[0]):
        for value in range(0, rawData.shape[1]):
            if(rawData[row][value] == 0):
                rawData[row][value] = rawData[row-1][value]
    data = rawData.tolist()
    reshapedData = []
    for index in range(len(data) - sequenceLength):
        reshapedData.append(data[index: index + sequenceLength])
    d0 = np.array(reshapedData)
    dr = np.zeros_like(d0)
    dr[:,1:,:] = d0[:,1:,:] / d0[:,0:1,:] - 1
    start = 2400
    end = int(dr.shape[0] + 1)
    unnormalized_bases = d0[start:end,0:1,20]
    
    #Splitting data set into training (First 90% of data points) and testing data (last 10% of data points)
    splitLine = round(0.9 * dr.shape[0])
    trainingData = dr[:int(splitLine), :]
    
    #Shuffle the data
    np.random.shuffle(trainingData)
    
    xTrain = trainingData[:, :-1]
    yTrain = trainingData[:, -1]
    yTrain = yTrain[:, 20]
    
    xTest = dr[int(splitLine):, :-1]
    yTest = dr[int(splitLine):, 49, :]
    yTest = yTest[:, 20]

    yDayBefore = dr[int(splitLine):, 48, :]
    yDayBefore = yDayBefore[:, 20]
    
    sequenceLength = sequenceLength
    windowSize = sequenceLength - 1 
    
    return xTrain, yTrain, xTest, yTest, yDayBefore, unnormalized_bases, windowSize


def createModel(windowSize, dropoutValue, activationFunction, lossFunction, optimizer):
    model = Sequential()
    model.add(Bidirectional(LSTM(windowSize, return_sequences=True), input_shape=(windowSize, xTrain.shape[-1]),))
    model.add(Dropout(dropoutValue))
    model.add(Bidirectional(LSTM((windowSize*2), return_sequences=True)))
    model.add(Dropout(dropoutValue))
    model.add(Bidirectional(LSTM(windowSize, return_sequences=False)))
    model.add(Dense(units=1))
    model.add(Activation(activationFunction))
    model.compile(loss=lossFunction, optimizer=optimizer)
    
    return model


def fitModel(model, xTrain, yTrain, batchNum, epochNum, splitValue):
    start = time.time()
    model.fit(xTrain, yTrain, batch_size= batchNum, nb_epoch=epochNum, validation_split= splitValue)
    trainingTime = int(math.floor(time.time() - start))
    modelJson = model.to_json()
    with open("model.json", "w") as jsonFile:
        jsonFile.write(modelJson)
    model.save_weights("model.h5")
    print("Model is saved to disk ...")
    return trainingTime


def loadModel():
    jsonFile = open('model.json', 'r')
    loadedModelJson = jsonFile.read()
    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    loadedModel.load_weights("model.h5")
    print("The model is loaded from disk ...")
    return loadedModel


def testModel(model, xTest, yTest, unnormalized_bases):
    yPredict = model.predict(xTest)
    real_yTest = np.zeros_like(yTest)
    real_yPredict = np.zeros_like(yPredict)
    for i in range(yTest.shape[0]):
        y = yTest[i]
        predict = yPredict[i]
        real_yTest[i] = (y+1)*unnormalized_bases[i]
        real_yPredict[i] = (predict+1)*unnormalized_bases[i]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_title("Bitcoin Price Over Time")
    plt.plot(real_yPredict, color = 'green', label = 'Predicted Price')
    plt.plot(real_yTest, color = 'red', label = 'Real Price')
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time (Days)")
    ax.legend()
    
    return yPredict, real_yTest, real_yPredict, fig


def price_change(yDayBefore, yTest, yPredict):
    yDayBefore = np.reshape(yDayBefore, (-1, 1))
    yTest = np.reshape(yTest, (-1, 1))
    deltaPredict = (yPredict - yDayBefore) / (1+yDayBefore)
    deltaReal = (yTest - yDayBefore) / (1+yDayBefore)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Percent Change in Bitcoin Price Per Day")
    plt.plot(deltaPredict, color='green', label = 'Predicted Percent Change')
    plt.plot(deltaReal, color='red', label = 'Real Percent Change')
    plt.ylabel("Percent Change")
    plt.xlabel("Time (Days)")
    ax.legend()
    plt.show()
    
    return yDayBefore, yTest, deltaPredict, deltaReal, fig


def binaryPrices(deltaPredict, deltaReal):
    deltaPredictBinary = np.empty(deltaPredict.shape)
    deltaRealBinary = np.empty(deltaReal.shape)

    #If the change in price is greater than zero, store it as a 1
    #If the change in price is less than zero, store it as a 0
    for i in range(deltaPredict.shape[0]):
        if deltaPredict[i][0] > 0:
            deltaPredictBinary[i][0] = 1
        else:
            deltaPredictBinary[i][0] = 0
    for i in range(deltaReal.shape[0]):
        if deltaReal[i][0] > 0:
            deltaRealBinary[i][0] = 1
        else:
            deltaRealBinary[i][0] = 0    

    return deltaPredictBinary, deltaRealBinary

def findPositiveNegatives(deltaPredictBinary, deltaRealBinary):
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    for i in range(deltaRealBinary.shape[0]):
        real = deltaRealBinary[i][0]
        predicted = deltaPredictBinary[i][0]
        if real == 1:
            if predicted == 1:
                truePositive += 1
            else:
                falseNegative += 1
        elif real == 0:
            if predicted == 0:
                trueNegative += 1
            else:
                falsePositive += 1
    return truePositive, falsePositive, trueNegative, falseNegative


def showStats(truePositive, falsePositive, trueNegative, falseNegative, yPredict, yTest):
    precision = float(truePositive) / (truePositive + falsePositive)
    recall = float(truePositive) / (truePositive + falseNegative)
    F1 = float(2 * precision * recall) / (precision + recall)
    MSE = mean_squared_error(yPredict.flatten(), yTest.flatten())

    return precision, recall, F1, MSE

def loadAndPrintData(dataFile):
    xTrain, yTrain, xTest, yTest, yDayBefore, unnormalized_bases, windowSize = loadData(dataFile, 50)
    print (xTrain.shape)
    print (yTrain.shape)
    print (xTest.shape)
    print (yTest.shape)
    print (yDayBefore.shape)
    print (unnormalized_bases.shape)
    print (windowSize)
    return xTrain, yTrain, xTest, yTest, yDayBefore, unnormalized_bases, windowSize

xTrain, yTrain, xTest, yTest, yDayBefore, unnormalized_bases, windowSize = loadAndPrintData("../../data/BTC Dataset for Py2.csv")

model = createModel(windowSize, 0.2, 'linear', 'mse', 'adam')
print (model.summary())
# trainingTime = fitModel(model, xTrain, yTrain, 1024, 100, .05)
model = loadModel()
yPredict, real_yTest, real_yPredict, fig1 = testModel(model, xTest, yTest, unnormalized_bases)

#Show the plot
plt.show(fig1)

#Print the training time
# print ("Training time", trainingTime, "seconds")
yDayBefore, yTest, deltaPredict, deltaReal, fig2 = price_change(yDayBefore, yTest, yPredict)

#Show the plot
plt.show(fig2)

# deltaPredictBinary, deltaRealBinary = binaryPrices(deltaPredict, deltaReal)

# print (deltaPredictBinary.shape)
# print (deltaRealBinary.shape)

# truePositive, falsePositive, trueNegative, falseNegative = findPositiveNegatives(deltaPredictBinary, deltaRealBinary)
# print ("True positives:", truePositive)
# print ("False positives:", falsePositive)
# print ("True negatives:", trueNegative)
# print ("False negatives:", falseNegative)

# precision, recall, F1, MSE = showStats(truePositive, falsePositive, trueNegative, falseNegative, yPredict, yTest)
# print ("Precision:", precision)
# print ("Recall:", recall)
# print ("F1 score:", F1)
# print ("Mean Squared Error:", MSE)