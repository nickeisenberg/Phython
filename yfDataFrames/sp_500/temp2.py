import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/sp_500/'

dirs = ['filt', 'unfilt']
OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']

dfs = {d: {t: [] for t in OHLCV} for d in dirs}
for d in dirs:
    path_ = os.path.join(path, d)
    for fn in os.listdir(path_):
        if fn.endswith('.csv'):
            df = pd.read_csv(os.path.join(path_, fn), index_col=0)
            # for col in df.columns.values:
            #     if np.isnan(max(df[col])):
            #         df.drop([col], axis=1, inplace=True)
        else:
            continue
        for t in OHLCV:
            if t in fn:
                # df.columns = pd.Series([f'SPY_{t}'])
                dfs[d][t].append(df)

for d in dirs:
    for t in OHLCV:
        print('--------------------------------------------------')
        print(d, t)
        for i in range(2):
            print(dfs[d][t][i].head())
            print(dfs[d][t][i].tail())

for x, d in enumerate(dirs):
    for y, t in enumerate(OHLCV):
        print('--------------------------------------------------')
        print(d, t)
        arrs_ = []
        times = []
        for i in range(2):
            arrs_.append(dfs[d][t][i].values.reshape(-1))
            times.append(dfs[d][t][i].index.values.reshape(-1))
        arr = np.hstack(arrs_)
        times = np.hstack(times)
        sorted_inds = np.argsort(times)
        times = times[sorted_inds]
        arr = arr[~np.isnan(arr)][sorted_inds]
        dfs[d][t] = pd.DataFrame(
                data=arr,
                index=times,
                columns=[f'SPY_{t}'])

fig, ax = plt.subplots(2, 5)
for i, d in enumerate(dirs):
    for j, t in enumerate(OHLCV):
        ax[i, j].plot(dfs[d][t].values.reshape(-1))
plt.show()

for d in dirs:
    for t in OHLCV:
        print('')
        print(d, t)
        print(dfs[d][t].head())
        print(dfs[d][t].tail())

jan_dfs = {d: {t: [] for t in OHLCV} for d in dirs}
feb_dfs = {d: {t: [] for t in OHLCV} for d in dirs}
for d in dirs:
    for t in OHLCV:
        df = dfs[d][t]
        times = df.index.values
        jan_bool = (times >= '2023-01') & (times < '2023-02')
        feb_bool = (times >= '2023-02') & (times < '2023-03')
        jan_dfs[d][t] = df.loc[jan_bool]
        feb_dfs[d][t] = df.loc[feb_bool]

for d in dirs:
    for t in OHLCV:
        print('')
        print(d, t)
        print('jan')
        print(jan_dfs[d][t].head())
        print(jan_dfs[d][t].tail())
        print('feb')
        print(feb_dfs[d][t].head())
        print(feb_dfs[d][t].tail())

for d in dirs:
    for t in OHLCV:
        jan_dfs[d][t].to_csv(
                os.path.join(path, d, f'SPY_{t}_01.csv')
                )
        feb_dfs[d][t].to_csv(
                os.path.join(path, d, f'SPY_{t}_02.csv')
                )
