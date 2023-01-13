import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from black_scholes_funcs import *
import datetime as dt
from scipy.stats import pearsonr, norm
from copy import deepcopy
import yfinance as yf

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/amzn/'

amzn = pd.read_csv(path + 'dec2022_1m.csv', index_col=0)
dates_str = amzn.index.values    
dates_dt = [dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in dates_str]

last_week_bool = dates_str > '2022-12-25'

amzn_beg = amzn.loc[~last_week_bool]
dates_str = amzn_beg.index.values    
dates_dt = [dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in dates_str]

amzn_end = amzn.loc[last_week_bool]
dates_str_end = amzn_end.index.values    
dates_dt_end = [dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
                for s in dates_str_end]

# group inds by days of the week
day_inds = {}
for i in np.arange(0, 5):
    day_inds[i] = []

for ddt, dstr in zip(dates_dt, dates_str):
    day_inds[ddt.weekday()].append(dstr)

day_df = {}  # dictionary of dfs of prices based on days of the week
for d, ind in day_inds.items():
    day_df[d] = amzn_beg.loc[ind]

day_stats = {}
for day, df in day_df.items():
    day_stats[day] = [np.diff(np.log(df['Open'].values)).mean(),
                      np.diff(np.log(df['Open'].values)).std()]

day_inds_end = {}
for i in np.arange(0, 5):
    day_inds_end[i] = []

for ddt, dstr in zip(dates_dt_end, dates_str_end):
    day_inds_end[ddt.weekday()].append(dstr)

day_df_end = {}  # dictionary of dfs of prices based on days of the week
for d, ind in day_inds_end.items():
    day_df_end[d] = amzn_end.loc[ind]

day_stats_end = {}
for day, df in day_df_end.items():
    if day == 0:
        continue
    day_stats_end[day] = [np.diff(np.log(df['Open'].values)).mean(),
                          np.diff(np.log(df['Open'].values)).std()]

for k, v in day_stats.items():
    print(k)
    print(v)
    print('')

for k, v in day_stats_end.items():
    print(k)
    print(v)
    print('')

day_indicator = np.array([d.weekday() for d in dates_dt])
day_open = {}
for day in day_df.keys():
    day_bool = (day_indicator == day).astype(int)
    sep = np.diff(day_bool)
    inds = np.where(sep == -1)[0] + 1
    series_list = []
    for ind in inds:
        dt_end = dates_dt[ind]
        dt_beg = dt_end - dt.timedelta(hours=24)
        str_beg = dt_beg.strftime('%Y-%m-%d %H:%M:%S')
        try:
            beg_ind = np.where(dates_str == str_beg)[0][0]
            print(beg_ind)
        except:
            print(f'prob with day {day}')
         
bol = (day_indicator == 0).astype(int)
boll = np.diff(bol)
endlocs = np.where(boll == -1)[0] + 1
for loc in endlocs:
    dtt = dates_dt[loc] - dt.timedelta(hours=24)
    dtt = dtt.strftime('%Y-%m-%d %H:%M:%S')
    try:
        ind = np.where(dates_str == dtt)[0][0]
        print(ind)
        print(dates_str[ind])
        print(day_indicator[ind])
        print('')
    except:
       print('no monday this week') 
        




day_df_noralized = {}
for day, df in day_df.items():
    series_list = []
    for col in df.columns:
        vec = deepcopy(df[col])
        mu, std = vec.mean(), vec.std()
        vec -= mu
        vec /= std
        series_list.append(vec)
    day_df_noralized[day] = pd.DataFrame(series_list).T

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
week_inds[no_of_weeks] = np.arange(week_start_inds[-1], dates_str.shape[0])

week_df = {}  # dictionary of dfs of price based on week
for k, ind in week_inds.items():
    week_df[k] = amzn_beg.iloc[ind]

week_df_normalized = {}
for week, df in week_df.items():
    series_list = []
    for col in df.columns:
        vec = deepcopy(df[col])
        mu, std = vec.mean(), vec.std()
        vec -= mu
        vec /= std
        series_list.append(vec)
    week_df_normalized[week] = pd.DataFrame(series_list).T

fig, ax = plt.subplots(1, 2)
ax[0].plot(week_df[0]['Open'].values)
ax[1].plot(week_df_normalized[0]['Open'].values)
plt.show()
#--------------------------------------------------


opens = amzn_beg['Open'].values
log_opens = np.log(opens)
log_returns = np.diff(log_opens)


