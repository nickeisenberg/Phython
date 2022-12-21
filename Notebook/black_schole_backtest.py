import numpy as np
import pandas as pd
import polygon as pg
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm
import tensorflow as tf

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/SP500_1m_data/AMZN/'

amzn_df0 = pd.read_csv(path + '20221128_20221202_MH.csv', index_col=0)
amzn_df1 = pd.read_csv(path + '20221205_20221209_MH.csv', index_col=0)

amzn_price0 = amzn_df0['Open'].values[: -1]
amzn_price1 = amzn_df1['Open'].values[: -1]

amzn_price = np.hstack((amzn_price0, amzn_price1))

print(amzn_price.shape)
print(amzn_price0.shape)
print(amzn_price1.shape)

time = range(amzn_price.shape[0])


plt.plot(time[: amzn_price0.shape[0]], amzn_price0, c='red')
plt.plot(time[amzn_price0.shape[0]:], amzn_price1, c='blue')
plt.xlim(0, len(time))
plt.show()

option_client = pg.OptionsClient('wEnDK0VciFtxJ1kThvfJXbiSKbkbZhfK')
amzn_call = pg.build_option_symbol('AMZN',
                                   date(2022, 12, 2),
                                   'call',
                                   94,
                                   _format='tradier')

amzn_call_dic = option_client.get_daily_open_close(amzn_call,
                                                   date(2022, 11, 28))

amzn_call_dic_end = option_client.get_daily_open_close(
        amzn_call,
        date(2022, 12, 2))


amzn_call_open = amzn_call_dic['open']

amzn_call_final = amzn_call_dic_end['close']

def black_scholes_call(S0, K, T, s, r):

    g = (np.log(S0 / K) + (r + (.5 * s ** 2)) * T) / (s * np.sqrt(T))
    
    h = g - s * np.sqrt(T)

    return (S0 * norm.cdf(g)) - (K * np.exp(-r * T) * norm.cdf(h))

bs = black_scholes_call(100, 100, 5, .03, .03)


x = np.linspace(0, 20, 21)
print(x)
print(x[::3])
print(x[1::3])
print(x[2::3])

def returns(S):
    return np.diff(S) / S[: -1]

print(amzn_price0.shape)

partitions = tf.keras.utils.timeseries_dataset_from_array(
        amzn_price0,
        targets=None,
        sequence_length=487,
        sequence_stride=487,
        batch_size=None)

partitions = np.array([p for p in partitions])

print(partitions.shape)

part_returns = np.diff(partitions, axis=1) / partitions[:, : -1]

mean_returns = part_returns.mean(axis=1)

std_returns = part_returns.std(axis=1)

print(std_returns.mean())

bs_price = black_scholes_call(amzn_price1[0],
                              94,
                              4,
                              std_returns.mean(),
                              .00237)

print(amzn_price1[0])
print(bs_price)
print(amzn_call_open)
print(amzn_call_final)



