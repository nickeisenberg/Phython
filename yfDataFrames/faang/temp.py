import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go

path0 = '/Users/nickeisenberg/Desktop/fix/faang/'
path1 = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/faang/'

dtypes = ['Open', 'High', 'Low', 'Close', 'Volume']

type0 = ['filtered', 'unfiltered']
type1 = ['filt', 'unfilt']

faang = ['AAPL', 'AMZN', 'GOOG', 'META', 'NFLX']

dfs0 = {t: {dt: [] for dt in dtypes} for t in type0}
for t in type0:
    for fn in os.listdir(os.path.join(path0, t)):
        if fn.endswith('.csv') and 'jan' in fn:
            df = pd.read_csv(os.path.join(path0, t, fn), index_col=0)
            for dt in dtypes:
                if dt in fn:
                    dfs0[t][dt].append(df)

for t in type0:
    for dt in dtypes:
        df = pd.concat(dfs0[t][dt])
        df.rename({nm: f'{nm}_{dt}' for nm in faang}, inplace=True, axis=1)
        df.rename({nm: f'{nm}_{dt}' for nm in faang}, inplace=True)
        bol = df.index.values >= '2023-02-01'
        bol *= df.index.values < '2023-02-03'
        dfs0[t][dt] = df.loc[bol]

dfs1 = {t: {dt: [] for dt in dtypes} for t in type1}
for t in type1:
    for fn in os.listdir(os.path.join(path1, t)):
        if not fn.endswith('.csv'):
            continue
        if '_02' in fn:
            df = pd.read_csv(os.path.join(path1, t, fn), index_col=0)
            for dt in dtypes:
                if dt in fn:
                    dfs1[t][dt].append(df)

for t0, t1 in zip(type0, type1):
    for dt in dtypes:
        dfs1[t1][dt] = pd.concat(dfs1[t1][dt])
        dfs1[t1][dt] = pd.concat((dfs0[t0][dt], dfs1[t1][dt]))

for t in type1:
    for dt in dtypes:
        dfs1[t][dt].to_csv(
                os.path.join(path1, t, f'{dt}_2023_02_.csv')
                )

#--------------------------------------------------
print(dfs0['filtered']['Open'].head())
print(dfs0['filtered']['Open'].tail())

print(dfs1['filt']['Open'].head())
print(dfs1['filt']['Open'].tail())

for col in dfs1['filt']['Open'].columns:
    fig = go.Figure()
    _ = fig.add_trace(
            go.Scatter(
                x=dfs1['filt']['Open'].index,
                y=dfs1['filt']['Open'][col]))
    fig.show()
