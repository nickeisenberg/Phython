import praw
from stats import *
from reddit import *
import yfinance as yf

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

# get the dates of all the submissions
months = ['January', 'Feburary', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
month_nos = [str(i).zfill(2) for i in range(1, 13)]
months = dict({*zip(months, month_nos)})

dates = []
for sub in subs_coms_reps.keys():
    date = [sub.title.split()[6: 7][0][: 4],
            months[sub.title.split()[4: 5][0]],
            sub.title.split()[5: 6][0][: 2]]
    date = [str(d) for d in date if int(d) == int(str(d))]
    date = '-'.join(date)
    dates.append(date)
dates.sort()
for d in dates:
    print(d)

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

gme_df = yf.Ticker('GME').history(
        period='1y', interval='1d',)

gme_open = gme_df['Open']
gme_close = gme_df['Close']

for i in gme_open.index:
    print(i)

print(gme_df.columns)
