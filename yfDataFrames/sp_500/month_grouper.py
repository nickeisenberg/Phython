import pandas as pd
import os

dirs = ['filt', 'unfilt']
path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/sp_500/'
types = ['Open', 'High', 'Close', 'Volume', 'Low']

dfs_in_dir = {dir: {t: [] for t in types} for dir in dirs}
for dir in dirs:
    for fn in os.listdir(os.path.join(path, dir)):
        if fn.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, dir, fn), index_col=0)
            if '^SPX' in df.columns:
                df.drop('^SPX', axis=1, inplace=True)
        for t in types:
            if t in fn:
                if f'SPY {t}' in df.columns:
                    df.rename({f'SPY {t}': f'SPY_{t}'}, inplace=True, axis=1)
                dfs_in_dir[dir][t].append(df)

jan_dfs_in_dir = {dir: {t: [] for t in types} for dir in dirs}
for dir in dirs:
    for t in types:
        df = pd.concat(dfs_in_dir[dir][t]).sort_index()
        jan_bool = (df.index >= '2023-01-01') & (df.index < '2023-02-01')
        jan_dfs_in_dir[dir][t] = df.loc[jan_bool]

for dir in dirs:
    for t in types:
        print(dir, t)
        print(jan_dfs_in_dir[dir][t].head())
        print(jan_dfs_in_dir[dir][t].tail())

for t in types:
    jan_dfs_in_dir['filt'][t].to_csv(
            os.path.join(path, 'filt', f'{t}_2023_01.csv')
            )

for t in types:
    jan_dfs_in_dir['unfilt'][t].to_csv(
            os.path.join(path, 'unfilt', f'{t}_2023_01.csv')
            )

