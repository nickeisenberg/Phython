import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/faang/'
dirs = ['filt', 'unfilt']
types = ['Open', 'Close', 'Low', 'High', 'Volume']

dfs = {dir: {t: [] for t in types} for dir in dirs}
for dir in dirs:
    for csv in os.listdir(os.path.join(path, dir)):
        if csv.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, dir, csv), index_col=0)
            for t in types:
                if t in csv:
                    dfs[dir][t].append(df)

for dir in dirs:
    for t in types:
        dfs[dir][t] = pd.concat(dfs[dir][t])

for dir in dirs:
    for t in types:
        fig = go.Figure()
        for col in dfs[dir][t].columns.values:
            _ = fig.add_trace(
                    go.Scatter(
                        x=dfs[dir][t].index,
                        y=dfs[dir][t][col],
                        name=col)
                    )
        fig.update_layout(title={'text': f'{dir}/{t}'})
        fig.show()

