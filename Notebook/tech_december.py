import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/5min/'
df_open = 'Dec2022tech_Open.csv'
df_vol = 'Dec2022tech_Volume.csv'

df_open = pd.read_csv(path + df_open, index_col=0)
df_vol = pd.read_csv(path + df_vol, index_col=0)

tickers = df_open.columns.values

open_vals = df_open.values.T
vol_vals = df_vol.values.T

for open, ticker in zip(open_vals, tickers):
    if open.shape != open[~np.isnan(open)].shape:
        print(ticker)
    else:
        print(f'---{ticker}---')

for open in open_vals:
    plt.clf()
    plt.plot(open)
    plt.pause(.1)
plt.show()

for open, ticker in zip(open_vals, tickers):
    Mm = MinMaxScaler()
    open = Mm.fit_transform(open.reshape((-1, 1)))
    plt.plot(open, label=f'{ticker}')
plt.legend()
plt.show()


