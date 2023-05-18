import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import median_filter
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

class ticker_history:

    def __init__(self, tickers: list):
        self.tickers = list(tickers)

    def ohlcv(self, start: datetime, end: datetime):

        no_weeks = int(np.ceil(((end - start).days + 1) / 7))
        weeks = np.arange(0, no_weeks, 1)

        data = []
        for i in weeks:
           start_ = start + timedelta(days=7 * int(i))
           if i == weeks[-1]:
                end_ = end
           else:
                end_ = start + timedelta(days=6 * (int(i) + 1))
           tick_df = yf.download(
                tickers=self.tickers,
                start=start_,
                end=end_,
                interval='1m',
                prepost=True
            )
           tick_df.index = tick_df.index.astype('int64') // 10**9
           data.append(tick_df)

        types = ['Open', 'High', 'Low', 'Close', 'Volume']
        ticker_series_pre = {
            tick: {type_: [] for type_ in types} for tick in self.tickers
        }
        for tick in self.tickers:
            for type_ in types:
                for d in data:
                    if len(self.tickers) == 1:
                        ticker_series_pre[tick][type_].append(d[type_])
                    else:
                        ticker_series_pre[tick][type_].append(d[type_][tick])
        for tick in self.tickers:
            for type_ in types:
                series = pd.concat(ticker_series_pre[tick][type_])
                series.name = f'{tick}_{type_}'
                ticker_series_pre[tick][type_] = series

        ticker_series = deepcopy(ticker_series_pre)
        for tick in self.tickers:
            for k, v in ticker_series[tick].items():
                ticker_series[tick][k] = v.interpolate(axis=0)

        ticker_filt_series = deepcopy(ticker_series)
        for tick in self.tickers:
            for type_ in types:
                ticker_filt_series[tick][type_] = pd.Series(
                    median_filter(
                        input=ticker_series[tick][type_],
                        size=5,
                        mode='nearest'),
                    index=ticker_filt_series[tick][type_].index,
                    name=ticker_filt_series[tick][type_].name)

        return ticker_series, ticker_filt_series

#--------------------------------------------------

# start = datetime(2023, 5, 1, 4 - 3, 0, 0)
# end = datetime(2023, 5, 6, 4 - 3, 0, 0)
# 
# faang = ['AMZN', 'META', 'GOOG', 'NFLX', 'AAPL']
# 
# history = ticker_history(['AMZN']).ohlcv(start, end)
# 
# df = history[1]['AMZN']['Open']
# 
# df

# df.index.astype('int64') // 10**9









