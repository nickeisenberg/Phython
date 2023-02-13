import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import median_filter
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

# Tickers to get info for
faang = ['AMZN', 'META', 'GOOG', 'NFLX', 'AAPL']
#--------------------------------------------------

# Get the info and interpolate the missing minute data
start = datetime(2023, 1, 16, 4 - 3, 0, 0)
end = start + timedelta(days=5, hours=16)

data = []
for i in range(4):
    start_ = start + timedelta(days=7 * i)
    end_ = end + timedelta(days=7 * i)
    data.append(
            yf.download(
                tickers=faang,
                start=start_,
                end=end_,
                interval='1m',
                prepost=True)
            )
#--------------------------------------------------

# Split up the multiindex so that the the dataframes can be easily stored
types = ['Open', 'High', 'Low', 'Close', 'Volume']
faang_dfs_pre = {}
for type_ in types:
    faang_dfs_pre[type_] = []
for type_ in types:
    for d in data:
        faang_dfs_pre[type_].append(d[type_])
for type_ in types:
    faang_dfs_pre[type_] = pd.concat(faang_dfs_pre[type_])

faang_dfs_pre['Open'].head()
faang_dfs_pre['Open'].tail()
#--------------------------------------------------

# Figure out the max number of missing prices in a row
bad_inds = {}
for tick in faang:
    bad_inds[tick] = []

nan_in_a_row = {}
for tick in faang:
    xxx = faang_dfs_pre['Open'][tick].values
    index = np.linspace(0, xxx.shape[0] - 1, xxx.shape[0])
    isnan = index[np.isnan(xxx)]
    isnan_diff = np.diff(isnan)
    count = 0
    thresh = 14
    for i in range(isnan.shape[0] - thresh):
        ind = isnan[i]
        next = isnan[i: i + thresh]
        bol = []
        for j in range(thresh):
            bol.append(ind + j == next[j])
        if np.array(bol).sum() == thresh:
            count += 1
            bad_inds[tick].append(ind)
    nan_in_a_row[tick] = count

for k, v in nan_in_a_row.items():
    print(k)
    print(v)
    print('')

print(bad_inds['NFLX'])  # It seems the NaN's are predominately in PM or AH
#--------------------------------------------------

# interpolate the prices
faang_dfs = deepcopy(faang_dfs_pre)
for k, v in faang_dfs.items():
    faang_dfs[k] = v.interpolate(axis=0)
#--------------------------------------------------

# Filter out the spikes that appear. Not sure what caused the spikes.
# Probably an after hours or premarket market order. 
faang_filt_dfs = deepcopy(faang_dfs)
for type_ in types:
    for col in faang_dfs[type_].columns:
        faang_filt_dfs[type_][col] = median_filter(
                input=faang_filt_dfs[type_][col],
                size=4,
                mode='nearest')
#--------------------------------------------------

# plot and view
faang_scaled_dfs = deepcopy(faang_filt_dfs['Open'])
Mm = MinMaxScaler()
faang_scaled_dfs[faang_scaled_dfs.columns] = Mm.fit_transform(
        faang_scaled_dfs[faang_scaled_dfs.columns])

fig = go.Figure()
for tick in faang:
    _ = fig.add_trace(
            go.Scatter(# x=faang_scaled_dfs[tick].index,
                       y=faang_scaled_dfs[tick].values,
                       name=f'{tick}')
            )
fig.show()
#--------------------------------------------------


# Store the dataframes
path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/faang/'

for type_ in types:
    faang_dfs[type_].to_csv(f'{path}/unfiltered/faang_{type_}_jan.csv')

for type_ in types:
    faang_filt_dfs[type_].to_csv(f'{path}/filtered/faang_{type_}_jan.csv')
#--------------------------------------------------

# read the csv files, scale and plot them
reload = {}
for type_ in types:
    reload[type_] = pd.read_csv(
            f'{path}/filtered/faang_{type_}_jan.csv', index_col=0
            )

reload['Open'].index

scaled = deepcopy(reload['Open'])
Mm = MinMaxScaler()
scaled[scaled.columns] = Mm.fit_transform(scaled[scaled.columns])

fig = go.Figure()
for tick in faang:
    _ = fig.add_trace(
            go.Scatter(y=scaled[tick].values, name=f'{tick}')
            )
fig.show()

