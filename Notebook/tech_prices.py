import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

tech = ['AAPL',
        'MSFT',
        'GOOG',
        'AMZN',
        # 'TCEHY',
        'TSM',
        # 'CSCO',
        # 'NFLX',
        # 'IBM',
        # 'SONY',
        # 'PYPL',
        # 'ABNB',
        # 'UBER',
        'TSLA',
        'NVDA',
        'META',
        'AMD',
        # 'ZG',
        # 'YELP',
        ]

data = []

for i in range(4):
    start = datetime(2022, 12, 19, 4) - timedelta(days=(i * 7), hours=3)
    end = start + timedelta(days=5, hours=16)
    data.append(
            yf.download(tech,
                        start=start,
                        end=end,
                        interval='15m',
                        prepost=True).iloc[:-1]
            )

df = pd.concat(data[::-1])

print(df.head())
print(df.tail())

column_names = []
for col in df.columns.values:
    if col[0] not in column_names:
        column_names.append(col[0])


path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/5min/'

for col in column_names:
    df[col].to_csv(f'{path}Dec2022tech_{col}.csv')


