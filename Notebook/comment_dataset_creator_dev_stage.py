import os
import pandas as pd
import praw
import yfinance as yf
from reddit import *

client_id = 'NA7yHLvDpSwcAEA1nmibNQ'
client_secret = 'qq_Odz0Vq6vIOPziOz2JYnoUJkMPog'
user_agent = 'phython_act'

# open up the subreddit
wsb = praw.Reddit(client_id=client_id,
                  client_secret=client_secret,
                  user_agent=user_agent).subreddit(
                          'wallstreetbets')

# get the daily discussion submissions
submissions = submission_getter(subreddit=wsb,
                                search='Daily Discussion',
                                no_of_submissions=4)

# get the comments
comments = comment_getter(submission_list=submissions,
                          no_of_comments=3)

# get the replies to the comments
subs_coms_reps = comment_replies(submission_list=submissions,
                                 submission_comments=comments,
                                 no_of_replies=10)

# preview the comments and replies
for sub in subs_coms_reps.keys():
    print('--------------------')
    print(sub.title)
    print('--------------------')
    for com in subs_coms_reps[sub].keys():
        print('----------comment----------')
        print(com.body)
        for rep in subs_coms_reps[sub][com]:
            print('----------reply----------')
            print(rep.body)
 
# get the dates of all the submissions
months = ['January', 'Feburary', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
month_nos = [str(i).zfill(2) for i in range(1, 13)]
months = dict({*zip(months, month_nos)})

dates = []
for sub in subs_coms_reps.keys():
# for sub in submissions:
    date = [sub.title.split()[6: 7][0][: 4],
            months[sub.title.split()[4: 5][0]],
            sub.title.split()[5: 6][0][: 2]]
    date = [str(d) for d in date if int(d) == int(str(d))]
    date = '-'.join(date)
    dates.append(date)
dates.sort()
for d in dates[-10:]:
    print(d)

# store comments and replies
path = '/Users/nickeisenberg/GitRepos/Phython/Notebook/Func_Outputs/gme_comments_replies'
for sub in subs_coms_reps.keys():
    date = [sub.title.split()[6: 7][0][: 4],
            months[sub.title.split()[4: 5][0]],
            sub.title.split()[5: 6][0][: 2]]
    date = [str(d) for d in date if int(d) == int(str(d))]
    date = '-'.join(date)
    dir_path = f'{path}/{date}/{sub.title}'
    os.makedirs(dir_path)
    com_count = 1
    for com in subs_coms_reps[sub].keys():
        com_path = f'{dir_path}/com_{com_count}.txt'
        open(com_path, 'x')
        with open(com_path, 'w') as fc:
            fc.write(com.body)
        rep_count = 1
        for rep in subs_coms_reps[sub][com]:
            rep_path = f'{dir_path}/com_{com_count}_rep_{rep_count}.txt'
            open(rep_path, 'x')
            with open(rep_path, 'w') as fr:
                fr.write(rep.body)
            rep_count += 1
        com_count += 1

# historical stock data
# create a meme index
memes = ['GME']
memes_history = []
for meme in memes:
    meme_df = yf.Ticker('GME').history(
            period='1y', interval='1d',)
    memes_history.append((meme, meme_df))
memes_history = dict(memes_history)

memes_open = []
memes_close = []
for meme in memes_history.keys():
    memes_open.append(memes_history[meme]['Open'])
    memes_close.append(memes_history[meme]['Close'])
memes_open = sum(memes_open) / len(memes_open)
memes_close = sum(memes_close) / len(memes_close)

memes_pchange_o2c = (memes_close - memes_open) / memes_open

data = ((memes_close[1:].values - memes_close[: -1].values) /
        memes_close[: -1].values)
index = memes_close[1:].index
memes_pchange_c2c = pd.Series(
        data=data,
        index=index)
