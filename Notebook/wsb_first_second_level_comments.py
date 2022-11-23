import praw
from datetime import datetime
from sys import exit 


reddit_read_only = praw.Reddit(client_id=client_id
                               client_secret=client_secret,
                               user_agent=user_agent)

subreddit = reddit_read_only.subreddit('wallstreetbets')
print(type(subreddit))
exit()

print(subreddit.display_name)
print('--------------------------------------------------')

submissions = subreddit.search('Discussion')
for submission in submissions:
    submission = submission
    break
print(submission.title)

top_comments = submission.comments
print(type(top_comments))
count = 0
for tc in top_comments:
    tc = tc
    count += 1
    if count ==2:
        break
print(f'top comment:\n{tc.body}')
print(f'author:\n{tc.author}\n')

count = 0
replies_tc = tc.replies
replies_tc.replace_more(limit=None)
for reply in replies_tc:
    print(reply.body)
    print(reply.author)
    print('')






