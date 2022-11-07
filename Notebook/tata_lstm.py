import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import exit
import yfinance as yf

#--------------------------------------------------
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
dataset_train = pd.read_csv(url)[::-1].reset_index(drop=True)
training_set = dataset_train.iloc[:, 1:2].values
#--------------------------------------------------
tata_info = yf.Ticker('TATACONSUM.NS')
tata_df = tata_info.history(period='max',
                            interval='1d',
                            actions=False).reset_index(drop=False)

date_start = dataset_train.index.values[0]
date_end = dataset_train.index.values[-1]
ind_start = tata_df.loc[tata_df['Date'] == date_start].index.values[0]
ind_end = tata_df.loc[tata_df['Date'] == date_end].index.values[0]

dataset_train = tata_df.iloc[ind_start:ind_end+1]
dataset_train = tata_dfsub.set_index('Date')
#--------------------------------------------------

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train =[]

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60 : i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

'''
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

model.fit(X_train, y_train, epochs=100, batch_size=32)
'''

# Test set
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
dataset_test = pd.read_csv(url)[::-1].reset_index(drop=True)
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0).reset_index(drop=True)

# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 : ]
# print(inputs.iloc[-4:])

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 : ].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 60 + len(dataset_test)):
    X_test.append(inputs[i - 60 : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='black', label='TATA Stock Price')
plt.plot(predicted_stock_price, color ='green', label='Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()

