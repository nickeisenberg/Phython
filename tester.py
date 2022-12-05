import praw
from stats import *
from reddit import *

client_id = 'JkwBK3M4E5CfK11a8oKkcw'
client_secret = 'llrNKLscK7Wcu8Fg28uoNsw5-I2LMw'
user_agent = 'psssat'

wsb = praw.Reddit(client_id=client_id,
                  client_secret=client_secret,
                  user_agent=user_agent).subreddit(
                          'wallstreetbets')

submissions = submission_getter(subreddit=wsb,
                                search='Discussion',
                                no_of_submissions=1)

comments = comment_getter(submission_list=submissions,
                          no_of_comments=3)

subs_coms_reps = comment_replies(submission_list=submissions,
                                 submission_comments=comments,
                                 no_of_replies=10)

for sub in subs_coms_reps.keys():
    print(sub.title)
    for com in subs_coms_reps[sub].keys():
        print(com.body)
        for rep in subs_coms_reps[sub][com]:
            print(rep.body)

