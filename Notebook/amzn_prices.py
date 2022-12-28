import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

start = datetime(2022, 11, 28, 4 - 3) + timedelta(days=7)
end = start + timedelta(days=5, hours=24)

dfs = []

for i in range(4):
    start = datetime(2022, 11, 28, 4 - 3) + timedelta(days=7 * i)
    end = start + timedelta(days=5, hours=24)
    amzn = yf.download('AMZN',
                       start=start,
                       end=end,
                       interval='1m',
                       prepost=True)
    dfs.append(amzn)

df = pd.concat(dfs)

print(df.head())
print(df.tail())

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/'

df.to_csv(path + 'amzn_december_1m.csv')
