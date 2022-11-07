import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

model = load_model('/Users/nickeisenberg/GitRepos/Python_Misc/MinimalWorkingExamples/Models/gme_lstm')

# gme_info = yf.Ticker('GME')
# gme_df = gme_info.history(period='2y',
#                           interval='1h',
#                           actions=False)

gme_df = pd.read_csv('/Users/nickeisenberg/GitRepos/Python_Misc/MinimalWorkingExamples/DataSets/gme_df.csv')
gme_open = gme_df['Open'].values

# Data was trained on gme_open[:2500]
# We will test the model on gme_open[2500:]

Mm = MinMaxScaler(feature_range=(0,1))
gme_test = gme_open[2500:]
gme_test = gme_test.reshape((gme_test.shape[0],1))

inputs_test = gme_open[len(gme_open) - len(gme_test) - 60 : ]
inputs_test = inputs_test.reshape(-1, 1)
inputs_test = Mm.fit_transform(inputs_test)

X_test = []
for i in range(60, 60 + len(gme_test)):
    X_test.append(inputs_test[i - 60 : i, 0])
X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)
predicted_price = Mm.inverse_transform(predicted_price)

plt.subplot(121)
plt.plot(gme_test, label='real')
plt.plot(predicted_price, label='predicted')
plt.legend()


def future_prediction(data=None, model=None, pred_amt=None, lookback=None):
    if len(data) < lookback:
        print('To big of a lookback')
        return None

    from sklearn.preprocessing import MinMaxScaler
    Mm = MinMaxScaler(feature_range=(0,1))
    data_scaled = Mm.fit_transform(data.reshape((-1,1))).T[0]

    pred_vals = []
    for i in range(pred_amt):
        if -lookback + i <= -1:
            data_lb = -lookback + i
            pred_lb = i
            inp_data = np.array(data_scaled[data_lb :])
            inp_pred = np.array(pred_vals[ : i])
            inp = np.hstack((inp_data, inp_pred))
            # print(inp_data)
            # print(inp_pred)
            # print(inp)
            # print(inp.reshape((1, len(inp), 1)))
            pred = model.predict(inp.reshape((1, len(inp), 1)))
            pred_vals.append(pred[0][0])

        if -lookback + i >= 0:
            inp = pred_vals[i - lookback : i]
            pred = model.predict(inp.reshape((1, len(inp), 1)))
            pred_vals.append(pred[0][0])

    pred_vals = Mm.inverse_transform(np.array(pred_vals).reshape((-1, 1)))
    return pred_vals.reshape(-1)



future_price = future_prediction(data=gme_open, model=model, pred_amt=15, lookback=60)
gme = gme_open[int(len(gme_open) * .9) : ]
time = len(gme) + 15
time = np.linspace(0, 1, time)

plt.subplot(122)
plt.plot(time[ : len(gme)], gme)
plt.plot(time[-15 : ], future_price)

plt.show()

