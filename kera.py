import keys
import datetime
from binance.client import Client 
import pandas as pd
import matplotlib.pylab as plot
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense 
from keras.layers import LSTM
import pickle

client = Client(keys.APIKey, keys.SecretKey)
symbol = 'BTCUSDT'
BTC = client.get_historical_klines(symbol = symbol, interval = Client.KLINE_INTERVAL_30MINUTE, start_str='1 year ago UTC')
# print (BTC)

BTC = pd.DataFrame(BTC, columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
# print (BTC)
BTC['Open time'] = pd.to_datetime(BTC['Open time'], unit = 'ms')
BTC.set_index('Open time', inplace = True)

BTC['Close'] = BTC['Close'].astype(float)
# plot.plot(BTC['Close'])
# plot.show()

data = BTC.iloc[:, 3:4].astype(float).values
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

training_set = data[:100]
test_set = data[100:]

x_train = training_set[0:len(training_set) -1]
y_train = training_set[1:len(training_set)]

x_test = test_set[0:len(test_set) -1]
y_test = test_set[1:len(test_set)]

x_train = np.reshape(x_train, (len(x_train), 1, x_train.shape[1]))
x_test = np.reshape(x_test, (len(x_test), 1, x_test.shape[1]))

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=16, shuffle=False)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

predicted_price = loaded_model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)
real_price = scaler.inverse_transform(y_test)
plot.plot(predicted_price, color='red', label='Predicted price of bitcoin')
plot.plot(real_price, color='blue', label='real price of bitcoin')
plot.title('Predicted vs. real price')
plot.legend(loc='best')
plot.show()






