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
sp_500 = ['SPY']
#--------------------------------------------------

start = datetime(2023, 2, 3, 4 - 3, 0, 0)
end = datetime(2023, 3, 1, 4 - 3, 0, 0)

no_weeks = np.arange(1, (end - start).days + 1, 1)[::7].size

data = []
for i in np.arange(0, no_weeks, 1):
    start_ = start + timedelta(days=7 * int(i))
    if i == np.arange(0, no_weeks, 1)[-1]:
        end_ = end
    else:
        end_ = start + timedelta(days=7 * (int(i) + 1))
    data.append(
            yf.download(
                tickers=sp_500,
                start=start_,
                end=end_,
                interval='1m',
                prepost=True)
            )

# Get the info and interpolate the missing minute data
#--------------------------------------------------

# Split up the multiindex so that the the dataframes can be easily stored
types = ['Open', 'High', 'Low', 'Close', 'Volume']
sp_500_dfs_pre = {type_: [] for type_ in types}
for type_ in types:
    for d in data:
        sp_500_dfs_pre[type_].append(d[type_])
for type_ in types:
    df = pd.concat(sp_500_dfs_pre[type_])
    df.rename(f'SPY_{type_}', inplace=True)
    sp_500_dfs_pre[type_] = df

#--------------------------------------------------

# Figure out the max number of missing prices in a row
bad_inds = {}
for tick in sp_500:
    bad_inds[tick] = []

sp_500_dfs_pre['Open'].head()
sp_500_dfs_pre['Open'].tail()

nan_in_a_row = {}
for tick in sp_500:
    # xxx = sp_500_dfs_pre['Open'][tick].values
    xxx = sp_500_dfs_pre['Open'].values
    index = np.linspace(0, xxx.shape[0] - 1, xxx.shape[0])
    isnan = index[np.isnan(xxx)]
    isnan_diff = np.diff(isnan)
    count = 0
    thresh = 2 
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
#--------------------------------------------------

# interpolate the prices
sp_500_dfs = deepcopy(sp_500_dfs_pre)

fig = go.Figure()
_ = fig.add_trace(
        go.Scatter(
            # x=sp_500_dfs['Open'].index.values,
            y=sp_500_dfs['Open'].values,
            )
        )
fig.show()

# for k, v in sp_500_dfs.items():  # SPY had no missing values
#     sp_500_dfs[k] = v.interpolate(axis=0)
#--------------------------------------------------

# Filter out the spikes that appear. Not sure what caused the spikes.
# Probably an after hours or premarket market order. 
sp_500_filt_dfs = deepcopy(sp_500_dfs)

for type_ in types:
    sp_500_filt_dfs[type_] = pd.Series(
           data=median_filter(
               input=sp_500_filt_dfs[type_],
               size=3,
               mode='nearest'),
           index=sp_500_dfs['Open'].index,
           name=type_)
#--------------------------------------------------

# plot and view
fig = go.Figure()
for tick in sp_500:
    _ = fig.add_trace(
            go.Scatter(# x=sp_500_scaled_dfs[tick].index,
                       y=sp_500_filt_dfs['Open'].values,
                       name=f'{tick}')
            )
    _ = fig.add_trace(
            go.Scatter(# x=sp_500_scaled_dfs[tick].index,
                       y=sp_500_dfs['Open'].values,
                       name=f'{tick}')
            )
fig.show()
#--------------------------------------------------


# Store the dataframes
path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/sp_500/'

for type_ in types:
    sp_500_dfs[type_].to_csv(f'{path}/unfilt/{type_}_2023_02.csv')

for type_ in types:
    sp_500_filt_dfs[type_].to_csv(f'{path}/filt/{type_}_2023_02.csv')
#--------------------------------------------------

# read the csv files, scale and plot them
reload = {}
for type_ in types:
    reload[type_] = pd.read_csv(
            f'{path}/filtered/sp_500_{type_}_jan.csv', index_col=0
            )

reload['Open'].tail()

scaled = deepcopy(reload['Open'])

Mm = MinMaxScaler()
scaled['Open'] = Mm.fit_transform(scaled['Open'].values.reshape((-1, 1)))

fig = go.Figure()
for tick in sp_500:
    _ = fig.add_trace(
            go.Scatter(y=scaled['Open'].values, name=f'{tick}')
            )
fig.show()

