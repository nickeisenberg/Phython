import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import minimize
import datetime as dt

path_to_amzn = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/amzn/'

amzn_dec = pd.read_csv(path_to_amzn + 'dec2022_1m.csv', index_col=0)

opens = amzn_dec['Open'].values
time = amzn_dec.index.values

log_opens = np.log(opens)
log_returns = np.diff(log_opens)

mu = log_returns.mean()
std = log_returns.std()

print(std)

norm_rv = norm(mu, std)

p_0 = norm_rv.ppf(1e-10)
p_1 = norm_rv.ppf(1 - 1e-10)
pdf_support = np.linspace(p_0, p_1, 1000)

hist, bins = np.histogram(log_returns, bins=1000)

fig = go.Figure()
_ = fig.add_trace(go.Histogram(x=log_returns, nbinsx=1000,
                               name='log returns'))
_ = fig.add_trace(go.Scatter(x=pdf_support,
                              y=norm_rv.pdf(pdf_support),
                             name=f'Normal pdf',
                             line={'width': 5}))
_ = fig.update_layout(
        title={'text': 'log returns of AMZN December 2022'}
        )
fig.show()

start = amzn_dec.index.values[0]

start_dt = dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
end_dt = start_dt + dt.timedelta(days=5, hours=16)

end = end_dt.strftime('%Y-%m-%d %H:%M:%S')

amzn_week1 = amzn_dec.loc[start: end]

week1_open = amzn_week1['Open'].values

week1_log_returns = np.diff(np.log(week1_open))

mu_week1 = week1_log_returns.mean()
std_week1 = week1_log_returns.std()

norm_rv_week1 = norm(mu_week1, std_week1)

fig = go.Figure()
_ = fig.add_trace(go.Histogram(x=week1_log_returns, nbinsx=1000))
_ = fig.add_trace(go.Scatter(x=pdf_support, y=norm_rv_week1.pdf(pdf_support)))
_ = fig.update_layout(
        title={'text': 'week one of December log returns'}
        )
fig.show()
