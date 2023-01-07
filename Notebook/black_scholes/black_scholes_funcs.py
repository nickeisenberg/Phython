import numpy as np
import polygon as pg
from datetime import date, datetime, timedelta
from scipy.stats import norm
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sys import exit

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
    with tf.GradientTape() as g:
        g.watch(S)
        y = f(S, T, K, s, r)
    return tf.keras.backend.get_value(g.gradient(y, S))

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
def geo_b_motion(T, S0, mu, sigma):
    delta = T[1] - T[0]
    dB = np.sqrt(delta) * np.random.normal(0, 1, T.shape[0] - 1)
    B = np.cumsum(dB)
    B = np.hstack((0, B))
    return S0 * np.exp((mu - sigma ** 2 / 2) * T + sigma * B)
#--------------------------------------------------

