import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from black_scholes_funcs import *
import datetime as dt
from scipy.stats import pearsonr, norm
from copy import deepcopy
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/amzn/'

amzn = pd.read_csv(path + 'dec2022_1m.csv', index_col=0)
dates_str = amzn.index.values    
dates_dt = [dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in dates_str]

last_week_bool = dates_str > '2022-12-25'

amzn_beg = amzn.loc[~last_week_bool]
dates_str_beg = amzn_beg.index.values    
dates_dt_beg = [dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in
                dates_str_beg]

amzn_end = amzn.loc[last_week_bool]
dates_str_end = amzn_end.index.values    
dates_dt_end = [dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
                for s in dates_str_end]
#--------------------------------------------------

# group the inds by weeks
day_ids = np.array([x.weekday() for x in dates_dt])
no_of_weeks = np.where(np.diff(day_ids) < -1)[0].shape[0] + 1
week_bool = np.diff(day_ids) >= 0

week_start_inds = np.hstack((np.zeros(1),
                             np.where(week_bool == False)[0] + 1)).astype(int)

week_inds = {}
count = 0
for ind1, ind0 in zip(week_start_inds[1:], week_start_inds[:-1]):
    week_inds[count] = np.arange(ind0, ind1)
    count += 1
week_inds[no_of_weeks - 1] = np.arange(week_start_inds[-1], dates_str.shape[0])

week_df = {}  # dictionary of dfs of price based on week
for k, ind in week_inds.items():
    week_df[k] = amzn.iloc[ind]

week_df_normalized = {}
for week, df in week_df.items():
    series_list = []
    for col in df.columns:
        vec = deepcopy(df[col])
        max, min = vec.max(), vec.min()
        vec -= min
        vec /= max 
        series_list.append(vec)
    week_df_normalized[week] = pd.DataFrame(series_list).T


fig, ax = plt.subplots(1, 2)
ax[0].plot(week_df[4]['Open'].values)
ax[1].plot(week_df_normalized[4]['Open'].values)
plt.show()

week_mu = {}
week_std = {}
for week, df in week_df.items():
    df = deepcopy(df.drop(labels='Volume', axis=1))
    log_ret = deepcopy(np.diff(np.log(df), axis=0))
    log_ret = pd.DataFrame(data=log_ret, columns=df.columns)
    week_mu[week] = log_ret.mean(axis=0)
    week_std[week] = log_ret.std(axis=0)
#--------------------------------------------------

# simulate the price based on previous week action
t_format = '%Y-%m-%d %H:%M:%S'
week_sims = {}
for week, df in week_df.items():
    if week == 0:
        continue
    week_sims[week] = []
    start = dt.datetime.strptime(df.index.values[0], t_format)
    end = dt.datetime.strptime(df.index.values[-1], t_format)
    t_delt = end - start
    mins = (t_delt.days + 1) * 16 * 60
    time = np.linspace(0, mins - 1, mins)
    delta = 1
    for i in range(20):
        sims = []
        for col in df.columns:
            if col == 'Volume':
                continue
            S0 = df[col].values[0]
            mu, std = week_mu[week - 1][col], week_std[week - 1][col]
            sims.append(geo_b_motion(time, delta, S0, mu, std))
        week_sims[week].append(pd.DataFrame(
            data=np.array(sims).T, columns=df.columns[:-1]))

titles = [f'week {i}' for i in range(1, 5)]
plts = [[1, 1], [1, 2], [2, 1], [2, 2]]
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=titles)
for week, sim in week_sims.items():
    _ = fig.add_trace(go.Scatter(y=week_df[week]['Open'].values,
                                 name=f'amzn {week}',
                                 line={'width': 4,
                                       'color': 'black'}),
                      row=plts[week - 1][0], col=plts[week - 1][1])
    for sim in week_sims[week]:
        _ = fig.add_trace(go.Scatter(y=sim['Open'].values,
                                     showlegend=False),
                      row=plts[week - 1][0], col=plts[week - 1][1])
_ = fig.update_layout(
        title={'text': 'simulations of AMZN for the last 4 weeks of December',
               'x': .5})
fig.show()


