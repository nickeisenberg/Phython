import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from black_scholes_funcs import *
from copy import deepcopy
from scipy.stats import pearsonr


path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/amzn/'
name = 'dec2022_1m'
amzn_df = pd.read_csv(f'{path}{name}.csv', index_col=0)

amzn_log_df = deepcopy(amzn_df)
amzn_log_df[amzn_log_df.columns[:-1]] = np.log(amzn_log_df[amzn_log_df.columns[:-1]])

# get the datetime and day of week (dow) for the timestamps
str_times = amzn_df.index.values
str_to_dt = np.array(
        [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S') for dt in str_times]
        )
str_to_dow = np.array(
        [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S').weekday() for dt in str_times]
        )

# yfinace missed some of the minutes of december
count = 0
for m, M in zip(str_to_dt[:-1], str_to_dt[1:]):
    m, M = m.minute, M.minute
    if M - m != 1 and M - m != -59:
        count += 1
        print(M - m)
print(count)

# parse out the individual weeks
week_identifier = str_to_dow[1:] - str_to_dow[:-1]
no_of_weeks = np.where(week_identifier == -4)[0].shape[0] + 2

# weeks are stored in weeks as 'week0', 'week1', ... etc 
weeks = {}
initial_day = str_to_dt[0]
for i in range(no_of_weeks):
    week_start = initial_day + timedelta(days=i * 7)
    week_end = week_start + timedelta(days=4, hours=15, minutes=59)
    weeks[f'week{i}'] = amzn_df.loc[str(week_start): str(week_end)]

weeks_log = {}
initial_day = str_to_dt[0]
for i in range(no_of_weeks):
    week_start = initial_day + timedelta(days=i * 7)
    week_end = week_start + timedelta(days=4, hours=15, minutes=59)
    weeks_log[f'week{i}'] = amzn_log_df.loc[str(week_start): str(week_end)]

# View the stock price
fig = make_subplots(specs=[[{'secondary_y': True}]])
_ = fig.add_trace(
        go.Scatter(x=time, y =amzn_df['Open'], name='Open Price'),
        secondary_y=False
        )
_ = fig.add_trace(
        go.Scatter(x=time, y=amzn_df['Volume'], name='Volume'),
        secondary_y=True,
        )
_ = fig.update_yaxes(title={'text': 'Price'},
                     secondary_y=False)
_ = fig.update_yaxes(title={'text': 'Volume'},
                     secondary_y=True)
fig.show()

# define black scholes variables
expire = datetime(2022, 12, 23, 16, 00, 00)  # friday of 'week3'
K = 83

amzn221230_C83_4 = call_price('AMZN',
                              K,
                              expire,
                              days_to_expire=4)
amzn221223_C83_0 = call_price('AMZN',
                              K,
                              expire,
                              days_to_expire=0)

prev_week = deepcopy(weeks['week2'])
prev_week_log = deepcopy(weeks_log['week2']['Open'].values) 
prev_week_log_returns = np.diff(prev_week_log) 

week_of_expire = deepcopy(weeks['week3'])

print(week_of_expire.head())
print(week_of_expire.tail())

print(week_of_expire.shape)

# PM = premarket, AH = after hours
# 1 time interval is from moday PM through friday AH
t_interval_in_min = 16 * 5 * 60
partition = np.linspace(0, 1, week_of_expire.shape[0])
delta = 1 / t_interval_in_min

# Under the geometric brownian motion assumption, the log returns are normally
# distributed with mean mu_sample = (mu - sigma ** 2 / 2) * delta and variance
# sigma_sample = sigma * sqrt(delta)
mu_sample = prev_week_log_returns.mean()
sigma_sample = prev_week_log_returns.std()
mu = mu_sample / delta + .5 * sigma_sample ** 2
sigma = sigma_sample * np.sqrt(1 / delta)

print(mu)
print(sigma)

S_0 = weeks['week3']['Open'].loc['2022-12-19 09:30:00']
rate = .00441 / 52 / 7 / 24 * t_interval_in_min
t_to_exp = (7.5 + 4 + 16 * 3 + 4.5 + 7.5) * 60 / t_interval_in_min
K = 83
fair_price = f(S_0, t_to_exp, K, sigma, rate)

print(fair_price)
print(amzn221230_C83_4)

S_0_pm = weeks['week3']['Open'].loc['2022-12-19 04:00:00']
sims = []
for i in range(100):
    sims.append(geo_b_motion(partition, delta, S_0_pm, mu, sigma))

for sim in sims:
    plt.plot(sim)
plt.plot(week_of_expire['Open'].values)
plt.show()

sim_corrs = []
for sim in sims:
    sim_corrs.append([pearsonr(sim, week_of_expire['Open'].values)[0], sim])

sim_corr_df = pd.DataFrame(
        data=[sim[1] for sim in sim_corrs],
        index=[sim[0] for sim in sim_corrs]
        )

sim_corr_df.sort_index(ascending=False, inplace=True)

print(sim_corr_df.index.values[:5])

plt.plot(week_of_expire['Open'].values)
plt.plot(sim_corr_df.iloc[0].values)
plt.show()

fig = make_subplots()
_ = fig.add_trace(
        go.Scatter(x=partition,
                   y =week_of_expire['Open'],
                   name='Open Price')
        )
for sim in sims:
    print(sim[0])
    _ = fig.add_trace(
            go.Scatter(x=partition,
                       y=sim,
                       name=f'Geometric brownian motion simulated price')
            )
_ = fig.update_yaxes(title={'text': 'Price'})
_ = fig.update_layout(
        title={
    'text': "A simultation of AMZN from 12-19-22 to 12-23-22",
    'x': .5
    },
        )
fig.show()
