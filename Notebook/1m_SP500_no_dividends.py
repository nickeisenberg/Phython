import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
import os

home_dir = '/Users/nickeisenberg/'
data_dir = 'GitRepos/Phython/yfDataFrames/SP500_1m_data/'
path = home_dir + data_dir

start = dt.datetime(2022, 11, 28)
end = dt.datetime(2022, 12, 2, 19, 59)

tickers = ['AMZN']

for ticker in tickers:
    t_path = path + ticker
    t_info = yf.Ticker(ticker)
    t_df = t_info.history(interval='1m',
                          start=start,
                          end=end,
                          prepost=True)
    t_start_date = f'{start.year}{start.strftime("%m")}{start.strftime("%d")}'
    t_end_date = f'{end.year}{end.strftime("%m")}{end.strftime("%d")}'
    t_df.to_csv(f'{t_path}/{t_start_date}_{t_end_date}.csv')

df_path = path + 'AMZN/20221128_20221202.csv'

df = pd.read_csv(df_path)

print(df.head())
print(df.tail())

