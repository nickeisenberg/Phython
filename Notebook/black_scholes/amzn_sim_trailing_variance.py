import numpy as np
import pandas as pd
import plotly.graph_objects as go
from black_scholes_funcs import *
from keras.utils import timeseries_dataset_from_array as tfa
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/amzn/'
csv_ = 'dec2022_1m.csv'
amzn = pd.read_csv(path + csv_, index_col=0)

time = amzn.index.values
data = amzn['Open'].values

# set the period and step in minutes
period_len = 20
step_size = 1
dataset = tfa(
        data=data,
        targets=None,
        sequence_length=period_len,
        sequence_stride=step_size,
        batch_size=None)

dataset = np.array([np.array(ts) for ts in dataset])

volatilities = np.diff(np.log(dataset), axis=1).std(axis=1)

fig = make_subplots(specs=[[{'secondary_y': True}]])
_ = fig.add_trace(
        go.Scatter(y=data, name='amzn'),
        secondary_y=False)
_ = fig.add_trace(
        go.Scatter(y=volatilities, name='volatility'),
        secondary_y=True)
_ = fig.update_layout(
        title={'text': f'AMZN versus volatility <br>{time[0]} --- {time[-1]}'}
        )
fig.show()

fig = make_subplots(rows=2, cols=1)
_ = fig.add_trace(
        go.Scatter(y=data, name='amzn'),
        row=1, col=1)
_ = fig.add_trace(
        go.Scatter(y=volatilities, name='volatility'),
        row=2, col=1)
_ = fig.update_layout(
        title={'text': f'AMZN versus volatility <br>{time[0]} --- {time[-1]}'}
        )
fig.show()

freqs_vol = np.fft.rfftfreq(int(volatilities.shape[0]), d=1 / volatilities.shape[0])
rfft_vol = abs(np.fft.rfft(volatilities))

fig = go.Figure()
_ = fig.add_trace(
        go.Scatter(x=freqs_vol[: 100],
                   y=rfft_vol[: 100])
        )
_ = fig.update_layout(
        title={'text': f'AMZN volatility frequencies\
                <br>{time[0]} --- {time[-1]}'}
        )
_ = fig.update_xaxes(
        title={'text': 'minutes'}
        )
fig.show()

freqs_price = np.fft.rfftfreq(int(data.shape[0]), d=1 / data.shape[0])
rfft_price = abs(np.fft.rfft(data))

fig = go.Figure()
_ = fig.add_trace(
        go.Scatter(x=freqs_price[1: 200],
                   y=rfft_price[1: 200])
        )
_ = fig.update_layout(
        title={'text': f'AMZN price frequencies\
                <br>{time[0]} --- {time[-1]}'}
        )
_ = fig.update_xaxes(
        title={'text': 'minutes'}
        )
fig.show()
