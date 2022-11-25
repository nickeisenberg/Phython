# Probably should turn this into a class
import praw


# type(subreddit) = praw.models.reddit.subreddit.Subreddit
# sort_by: 'hot', 'new', 'top', 'rising'
# If search != None and type(search) = str then that will override sort_by
def submission_getter(
    subreddit,
    sort_by='top',
    search=None,
    search_sort_by='relevance',
    no_of_submissions=10):

    if isinstance(search, str):
        submissions = subreddit.search(search, sort=search_sort_by)
    elif sort_by == 'top':
        submissions = subreddit.top(limit=no_of_submissions)
    elif sort_by == 'hot':
        submissions = subreddit.hot(limit=no_of_submissions)
    elif sort_by == 'new':
        submissions = subreddit.new(limit=no_of_submissions)
    elif sort_by == 'rising':
        submissions = subreddit.rising(limit=no_of_submissions)

    submission_list = [submission for submission in submissions]

    return submission_list


# comment_type:  for the mean time, this will sort comments by 'top'
def submission_to_comments_list(
    submission,
    no_of_comments=10):

    top_comments = submission.comments
    comments_list = []
    count = 1
    for tc in top_comments:
        comments_list.append(tc)
        if count == no_of_comments:
            break
        count += 1

    return comments_list


# Get the first level replies to comment in a CommentForest
# comment may either be a comment or a comments_list
def first_level_replies(comment):
    
    replies = comment.replies  # initiate the CommentForest
    replies.replace_more(limit=None)  # replace the MoreComments with Comments
    reply_list = []
    for reply in replies:
        reply_list.append(reply)

    return reply_list


if __name__ == '__main__':
    

    reddit_read_only = praw.Reddit(client_id=client_id,
                                   client_secret=client_secret,
                                   user_agent=user_agent)
    
    subreddit_name = 'wallstreetbets'
    subreddit = reddit_read_only.subreddit(subreddit_name) 
    
    count = 1
    for sub in subreddit.search('Discussion'):
        count += 1
        if count == 200:
            break
    print(count)
    exit()

    wsb_top_submission = submission_getter(subreddit=subreddit,
                                           search='Discussion',
                                           no_of_submissions=1)

    sub = wsb_top_submission[0]

    print(sub.title)

    sub_comment_list = submission_to_comments_list(submission=sub,
                                                   no_of_comments=3)
    
    path = '/Users/nickeisenberg/GitRepos/Phython/Notebook/Func_Outputs/'
    name = 'comments_replies2.txt'
    p_n = path + name

    count_c = 1
    with open(p_n, 'w') as f:
        for comment in sub_comment_list:
            print('--------------------')
            print(f'processing comment: {count_c} / {len(sub_comment_list)}')
            print('--------------------')
            f.write(f'Comment_{count_c}\n')
            f.write(f'{comment.body}\n')
            replies = first_level_replies(comment)
            count_r = 1
            for reply in replies:
                f.write(f'Reply_{count_r}\n')
                f.write(f'{reply.body}\n')
                count_r += 1
            count_c += 1
