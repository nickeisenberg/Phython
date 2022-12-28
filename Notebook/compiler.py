import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/'

df = pd.read_csv(path + 'amzn_december_1m.csv', index_col=0)

time = df.index.values
open = df['Open'].values.reshape(-1)

fig = go.Figure([go.Scatter(
    x=time, y=open)])
fig.show()

