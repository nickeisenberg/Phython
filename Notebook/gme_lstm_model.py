import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# gme_info = yf.Ticker('GME')
# gme_df = gme_info.history(period='2y',
#                           interval='1h',
#                           actions=False)

gme_df = pd.read_csv('/Users/nickeisenberg/GitRepos/Python_Misc/MinimalWorkingExamples/DataSets/gme_df.csv')
gme_open = gme_df['Open'].values

gme_train = gme_open[ : 2500]
gme_train = gme_train.reshape((gme_train.shape[0], 1))

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
gme_train_scaled = sc.fit_transform(gme_train)

X_train = []
y_train = []
for i in range(60, len(gme_train_scaled)):
    X_train.append(gme_train_scaled[i - 60 : i, 0])
    y_train.append(gme_train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=25, batch_size=32)

model.save('/Users/nickeisenberg/GitRepos/Python_Misc/MinimalWorkingExamples/Models/gme_lstm')

# Test data
gme_test = gme_open[2500:]
gme_test = gme_test.reshape((gme_test.shape[0],1))

inputs_test = gme_open[len(gme_open) - len(gme_test) - 60 : ]
inputs_test = inputs_test.reshape(-1, 1)
inputs_test = sc.fit_transform(inputs_test)

X_test = []
for i in range(60, 60 + len(gme_test)):
    X_test.append(inputs_test[i - 60 : i, 0])
X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# print(model.predict(X_test[-1].reshape(1, len(X_test[-1]), 1)))
# print(X_test[-1].reshape(1, len(X_test[-1]), 1))
predicted_price = model.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)

# plt.subplot(121)
plt.plot(gme_test, label='real')
plt.plot(predicted_price, label='predicted')
plt.legend()
plt.show()

