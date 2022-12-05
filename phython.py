import numpy as np
from sklearn.preprocessing import MinMaxScaler
import praw

# turn off scientific notation
np.set_printoptions(suppress=True)


#-statistics-functions---------------------------------------------------------

# need to remove the gmt offset that comes with yfinance date/time outputs
def historical_volitility(stock_price,
                          method=None,
                          s_ind=0,
                          e_ind=-1):

    stock_price = stock_price[s_ind: e_ind]

    dt = np.diff(stock_price)
    # mean_return = np.mean(dt)
    volitility = np.std(dt)

    return volitility


def pearson_corr(x, y):

    ux, uy = np.mean(x), np.mean(y)
    x_cen, y_cen = x - ux, y - uy

    pear_corr = (np.dot(x_cen, y_cen) /
                 (np.sqrt(np.sum(np.square(x_cen))) *
                  np.sqrt(np.sum(np.square(y_cen)))))

    return pear_corr


def norm_dot(x, y):

    n_dot = (np.dot(x, y) *
             (np.sqrt(np.sum(np.square(x))) *
              np.sqrt(np.sum(np.square(y)))))

    return n_dot


def find_corr(stock_price,
              pat_start=None,
              pat_end=None,
              standardize=True):

    stock_price = stock_price.reshape((-1, 1))

    if standardize:
        Mm = MinMaxScaler(feature_range=(0, 1))
        stock_price = Mm.fit_transform(stock_price)

    if isinstance(pat_start, int) and isinstance(pat_end, int):
        pat = stock_price[pat_start: pat_end]
        pat_len = len(pat)
        pat_ind = [pat_start, pat_end]

    corr_scores = []
    for i in range(0, pat_ind[0] + 1):
        ref = stock_price[i: i + pat_len]
        p_score = pearson_corr(pat.reshape(-1), ref.reshape(-1))
        corr_scores.append([i, p_score])
    corr_scores = np.array(corr_scores)
    corr_scores = corr_scores[corr_scores[:, 1].argsort()][::-1]

    top_scores = np.empty(0)
    for i in range(len(corr_scores[:, 0])):

        ref_ind = corr_scores[i][0]
        if np.abs(ref_ind - pat_ind[0]) < pat_len:
            continue

        if len(top_scores) == 0:
            top_scores = np.append(top_scores,
                                   [ref_ind, corr_scores[i][1]])
            top_scores = top_scores.reshape((1, 2))
            continue

        dist_i = np.min(np.abs(top_scores[:, 0] - ref_ind))
        if dist_i >= pat_len / 5:
            top_scores = np.vstack((top_scores,
                                    [ref_ind, corr_scores[i][1]]))

        if len(top_scores) == 9:
            break
    top_scores = np.array(top_scores)
    top_scores = top_scores

    return top_scores


#-Reddit-functions-------------------------------------------------------------

# type(subreddit) = praw.models.reddit.subreddit.Subreddit
# sort_by: 'hot', 'new', 'top', 'rising'
# If search != None and type(search) = str then that will override sort_by
def submission_getter(subreddit=None,
                      sort_by='top',
                      search=None,
                      search_sort_by='relevance',
                      no_of_submissions=10):

    print('starting submission getter')

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

    submission_list = []
    count = 1
    for sub in submissions:
        submission_list.append(sub)
        if count == no_of_submissions:
            break
        count += 1

    return submission_list


def comment_getter(submission_list=None,
                   no_of_comments=10):

    print('starting submissions to comments dic')

    submission_coms = {submission: [] for submission in submission_list}

    for submission in submission_coms.keys():
        print(submission.title)
        submission_coms[submission] = submission.comments[: no_of_comments]

    return submission_coms


def comment_replies(submission_list=None,
                    submission_comments=None,
                    no_of_replies=10):

    print('starting comments replies')

    submissions_comments_replies = {sub: {} for sub in submission_list}

    for sub in submission_list:
        comments_replies = {com: [] for com in submission_comments[sub]}
        count_c = 1
        for com in submission_comments[sub]:
            print(f'COMMENT {count_c}')
            replies = com.replies
            replies.replace_more(limit=None)
            replies = replies[: no_of_replies]
            for reply in replies:
                comments_replies[com].append(reply)
            count_c += 1

        submissions_comments_replies[sub] = comments_replies

    return submissions_comments_replies


if __name__ == '__main__':

    # test here

