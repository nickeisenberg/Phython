import praw
import pandas as pda
from sys import exit
from datetime import datetime

reddit_read_only = praw.Reddit(client_id='JkwBK3M4E5CfK11a8oKkcw',
                                 client_secret='llrNKLscK7Wcu8Fg28uoNsw5-I2LMw',
                                 user_agent='psssat')

subreddit = reddit_read_only.subreddit('climbing')

print(subreddit.display_name)
print('--------------------------------------------------')

post_count = 1
for submission in subreddit.hot(limit=3):
    
    if submission.title.startswith('Weekly'):

        print(f'Post {post_count}: {submission.title}')
        print('-------------------------')

        com_count = 1
        for tc in submission.comments:
            print(f'comment {com_count}: {tc.body}')
            print(f'tc type: {type(tc)}')
            print(f'author: {tc.author}')
            print(f'time: {datetime.fromtimestamp(tc.created_utc)}\n')
           
            reply_count = 1
            for reply in tc.replies:
                print(f'reply {reply_count}: {reply.body}\n')
                reply_count += 1

            com_count += 1
            if com_count == 3:
                break


    post_count += 1
    if post_count > 2:
        break
