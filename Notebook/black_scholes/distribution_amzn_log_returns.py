import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        title={'text': f'log returns of AMZN:\
                <br>{time[0]} - {time[-1]}'}
        )
fig.show()

start = amzn_dec.index.values[0]
start_dt = dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
end_dt = start_dt + dt.timedelta(days=5, hours=16)
end = end_dt.strftime('%Y-%m-%d %H:%M:%S')

t_format = '%Y-%m-%d %H:%M:%S'
week_data = {}
week_dates = {}
for i in range(4):
    w_start_dt = start_dt + dt.timedelta(days=7*i)
    w_end_dt = w_start_dt + dt.timedelta(days=5, hours=16)
    w_start_str = dt.datetime.strftime(w_start_dt, t_format)
    w_end_str = dt.datetime.strftime(w_end_dt, t_format)
    week_dates[i] = [w_start_str, w_end_str]
    week_data[i] = amzn_dec.loc[w_start_str: w_end_str]

week_log_returns = {}
for i in range(4):
    week_log_returns[i] = np.diff(np.log(week_data[i]['Open'].values))

def sub_plot_indicators(nrows, ncols):
    sub_plot_no = []
    for i in range(1, nrows * ncols + 1):
        sub_plot_no.append([int(np.floor(1 + ((i - 1) / ncols))),
                            int(i - ncols * np.floor((i - 1) / ncols))])
    return sub_plot_no

sub_p_titles = []
for i in range(4):
    sub_p_titles.append(f'{week_dates[i][0]} --- {week_dates[i][1]}')
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=sub_p_titles)
sub_p_nos = sub_plot_indicators(2, 2)
for i in range(4):
    data = week_log_returns[i]
    mu, std = data.mean(), data.std()
    print(mu, std)
    rv = norm(mu, std)
    pdf = rv.pdf
    support = np.linspace(rv.ppf(.0001), rv.ppf(.9999), 1000)
    _ = fig.add_trace(
            go.Histogram(x=data, nbinsx=100, showlegend=False),
            row=sub_p_nos[i][0], col=sub_p_nos[i][1])
    _ = fig.add_trace(
            go.Scatter(x=support, y=pdf(support), showlegend=False,
                       line={'width': 4,
                             'color': 'black'}),
            row=sub_p_nos[i][0], col=sub_p_nos[i][1])
fig.update_layout(
        title={'text': f'Amzn weekly log returns'})
fig.show()
