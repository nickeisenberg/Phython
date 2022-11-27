import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import praw

# turn off scientific notation
np.set_printoptions(suppress=True)


# need to remove the gmt offset that comes with yfinance date/time outputs
class Volitility:

    def __init__(self, ticker=None, data=None, data_path=None):
        self.ticker = ticker
        self.data_path = data_path
        self.data = data

    def historical(self,
                   method=None,
                   s_date=0,
                   e_date=-1,
                   period=None,
                   interval=None):

        data = self.data
        data_path = self.data_path
        ticker = self.ticker

        if isinstance(data_path, str):
            data = pd.read_csv(data_path)

        elif isinstance(data, pd.DataFrame):
            data = self.data

        else:
            data = yf.Ticker(ticker).history(period=period,
                                             interval=interval,
                                             # prepost=prepost,
                                             actions=False)

        if isinstance(s_date, str):
            s_date = data.index[data['Date'] == s_date].values[0]
        if isinstance(e_date, str):
            e_date = data.index[data['Date'] == e_date].values[0]

        data = data.iloc[s_date: e_date]

        if method == 'open':
            open_p = data['Open'].values
            self.dt = np.diff(open_p)
            self.mean_return = np.mean(self.dt)
            self.volitility = np.std(self.dt)

        elif method == 'close':
            close_p = data['Close'].values
            self.dt = np.diff(close_p)
            self.mean_return = np.mean(self.dt)
            self.volitility = np.std(self.dt)

        elif method == 'hl':
            high_p = data['High'].values
            low_p = data['Low'].values
            self.dt = high_p - low_p
            self.mean_return = np.mean(self.dt)
            self.volitility = np.std(self.dt)

        return np.std(self.dt)

    def implied(self):

        # find a way to get options data and black-scholes formula
        return None


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


class Correlation:

    def __init__(self, ticker=None, data_path=None):
        self.ticker = ticker
        self.data_path = data_path

    def find_corr(self,
                  period=None,
                  interval=None,
                  pat_start=None,
                  pat_end=None,
                  OHLC='Open',
                  prepost=False,
                  standardize=True):

        data_path = self.data_path
        ticker = self.ticker

        if isinstance(data_path, str):
            data = pd.read_csv(data_path)

        else:
            data = yf.Ticker(ticker).history(period=period,
                                             interval=interval,
                                             prepost=prepost,
                                             actions=False)

        data = data[OHLC].values.reshape((-1, 1))

        if standardize:
            Mm = MinMaxScaler(feature_range=(0, 1))
            data_scaled = Mm.fit_transform(data)

        if isinstance(pat_start, int) and isinstance(pat_end, int):
            pat = data[pat_start: pat_end]
            pat_s = data_scaled[pat_start: pat_end]
            pat_len = len(pat)
            pat_ind = [pat_start, pat_end]

        corr_scores = []
        for i in range(0, pat_ind[0] + 1):
            ref_s = data_scaled[i: i + pat_len]
            p_score_s = pearson_corr(pat_s.reshape(-1), ref_s.reshape(-1))
            corr_scores.append([i, p_score_s])
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
        self.top_scores = top_scores

        return top_scores


class reddit_phy:

    # def __init__(self,
    #              client_id=None,
    #              client_secret=None,
    #              user_agent=None,
    #              subreddit_name=None):

    #     self.client_id = client_id
    #     self.client_secret = client_secret
    #     self.user_agent = user_agent
    #     self.subreddit_name = subreddit_name
    #     self.subreddit = praw.Reddit(client_id=self.client_id,
    #                                  client_secret=self.client_secret,
    #                                  user_agent=self.user_agent).subreddit(
    #                                      self.subreddit_name)

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

        # self.submission_list = submission_list

        return submission_list

    def comment_getter(submission_list=None,
                       no_of_comments=10):

        print('starting submissions to comments dic')

        submission_coms = {submission: [] for submission in submission_list}

        for submission in submission_coms.keys():
            print(submission.title)
            submission_coms[submission] = submission.comments[: no_of_comments]

        # self.submission_comments = submission_comments

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

        # self.submissions_comments_replies = submissions_comments_replies

        return submissions_comments_replies


if __name__ == '__main__':

    client_id = 'JkwBK3M4E5CfK11a8oKkcw'                                              
    client_secret = 'llrNKLscK7Wcu8Fg28uoNsw5-I2LMw'
    user_agent = 'psssat'

    wsb = praw.Reddit(client_id=client_id,
                      client_secret=client_secret,
                      user_agent=user_agent).subreddit(
                          'wallstreetbets')

    submissions = reddit_phy.submission_getter(subreddit=wsb,
                                               search='Discussion',
                                               no_of_submissions=1)

    comments = reddit_phy.comment_getter(submission_list=submissions,
                                         no_of_comments=3)

    subs_coms_reps = reddit_phy.comment_replies(submission_list=submissions,
                                                submission_comments=comments,
                                                no_of_replies=10)

    for sub in subs_coms_reps.keys():
        print('--------------------')
        print(f'{sub.title}\n')
        print('--------------------\n')
        for com in subs_coms_reps[sub].keys():
            print(f'Comment:\n{com.body}\n')
            for rep in subs_coms_reps[sub][com]:
                print(f'Reply:\n{rep.body}')
