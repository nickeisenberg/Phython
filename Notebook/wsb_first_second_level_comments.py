import praw
from datetime import datetime
from sys import exit 

client_id = 'JkwBK3M4E5CfK11a8oKkcw'
client_secret = 'llrNKLscK7Wcu8Fg28uoNsw5-I2LMw'
user_agent = 'psssat'

reddit_read_only = praw.Reddit(client_id=client_id,
                               client_secret=client_secret,
                               user_agent=user_agent)

subreddit = reddit_read_only.subreddit('wallstreetbets')
print(subreddit.display_name)
print(type(subreddit))

# get the first 5 submissions
submissions = subreddit.search('Discussion')

sub_dic = {}
count = 1
for submission in submissions:
    sub_dic[f'sub_{count}'] = submission
    if count == 5:
        break
    count += 1

# get the first 5 top comments of the first submission
top_comments = sub_dic['sub_1'].comments

tc_dic = {}
count = 1
for tc in top_comments:
    print(f'progress tc: {count} / {5}')
    tc_dic[f'tc_{count}'] = tc
    if count == 5:
        break
    count += 1

# get all first level replies to the top comments
reply_dic = {}
count = 1
total = len(tc_dic.keys())
for name, tc in tc_dic.items():
    print(f'progress replies: {count} / {total}')
    replies_tc = tc.replies
    replies_tc.replace_more(limit=None)
    tc_no = name.split('_')[-1]
    reply_dic[f'tc_{tc_no}_replies'] = []
    for reply in replies_tc:
        reply_dic[f'tc_{tc_no}_replies'].append(reply.body)
    count += 1

for i in range(1, 6):
    tc = tc_dic[f'tc_{i}']
    replies = reply_dic[f'tc_{i}_replies']
    print(f'----------Comment-{i}----------:\n')
    print(f'{tc.body}\n')
    print(f'{type(tc.body)}\n')
    print(f'-----Replies-to-comment-{i}-----:\n')
    for k, r in enumerate(replies):
        print(f'---Reply-{k+1}---:\n')
        print(f'{r}\n')


