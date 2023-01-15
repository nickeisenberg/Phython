import re
import numpy as np
import polygon as pg
from datetime import timedelta
from scipy.stats import norm
from tensorflow import GradientTape
from keras.utils import timeseries_dataset_from_array
from keras.backend import get_value

# use polygon to get the option price
def call_price(ticker, strike, expire, days_to_expire):
    option_client = pg.OptionsClient('wEnDK0VciFtxJ1kThvfJXbiSKbkbZhfK')
    call = pg.build_option_symbol(ticker,
                                  expire,
                                  'call',
                                  strike,
                                  _format='tradier')
    call_dic = option_client.get_daily_open_close(
            call,
            expire - timedelta(days=days_to_expire)
            )
    return call_dic
#--------------------------------------------------

# parse through the stock prices
# specify the period
def  time_parser(time, delims='-|:|\s'):
    return re.split(delims, time)

#--------------------------------------------------

# trailing volatility calulator
def trailing_volatility(time_series, period_length, step_size=1):
    periods = timeseries_dataset_from_array(
            data=time_series,
            targets=None,
            sequence_length=period_length,
            sequence_stride=step_size,
            batch_size=None)
    period_volatilites = []
    for p in periods:
        period_volatilites.append(np.array(p).std())
    period_volatilites = np.array(period_volatilites)
    return period_volatilites

# formulize the black-scholes framework
# S = stock price
# T = time to expire
# K = strike
# s = stadard deviation of log(S_t)
# r = risk free rate
# B = bond price

def f(S, T, K, s, r):
    g = (np.log(S / K) + (r + (.5 * s ** 2)) * T) / (s * np.sqrt(T))
    h = g - s * np.sqrt(T)
    return (S * norm.cdf(g)) - (K * np.exp(-r * T) * norm.cdf(h))

def f_x(S, T, K, s, r):
    with GradientTape() as g:
        g.watch(S)
        y = f(S, T, K, s, r)
    return get_value(g.gradient(y, S))

def self_financing(t, B, S, T, K, s, r):
    a_t = f_x(S, T - t, K, s, r)
    b_t = (f(S, T - t, K, s, r) - (a_t * S)) / B
    return [a_t, b_t]
#--------------------------------------------------

# brownian and geometric brownian motion

# T is an array of time
def b_motion(T):
    delta = T[1] - T[0]
    dB = np.sqrt(delta) * np.random.normal(0, 1, T.shape[0] - 1)
    B = np.cumsum(dB)
    B = np.hstack((0, B))
    return B

# dS = mu St dt + sigma St dBt
def geo_b_motion(T, delta, S0, mu, sigma):
    dB = np.sqrt(delta) * np.random.normal(0, 1, T.shape[0] - 1)
    B = np.cumsum(dB)
    B = np.hstack((0, B))
    return S0 * np.exp((mu - sigma ** 2 / 2) * T + sigma * B)
#--------------------------------------------------

