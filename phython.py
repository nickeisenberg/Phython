import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import praw

# need to remove the gmt offset that comes with yfinance date/time outputs
class Volitility:

    def __init__(self, ticker=None, data=None, data_path=None):
        self.ticker = ticker
        self.data_path = data_path
        self.data = data

    def historical(self, method=None, s_date=0, e_date=-1, period=None, interval=None):

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

        data = data.iloc[s_date : e_date]

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
    return np.dot(x_cen, y_cen) / (np.sqrt(np.sum(np.square(x_cen))) * np.sqrt(np.sum(    np.square(y_cen))))

def norm_dot(x, y):
    return np.dot(x, y) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y))))

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

        data = data[OHLC].values.reshape((-1,1))

        if standardize:
            Mm = MinMaxScaler(feature_range=(0,1))
            data_scaled = Mm.fit_transform(data)

        if isinstance(pat_start, int) and isinstance(pat_end, int):
            pat = data[pat_start : pat_end]
            pat_s = data_scaled[pat_start : pat_end]
            pat_len = len(pat)
            pat_ind = [pat_start, pat_end]

        corr_scores = []
        for i in range(0, pat_ind[0] + 1):
            ref_s = data_scaled[i : i + pat_len]
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
                top_scores = np.append(top_scores, [ref_ind, corr_scores[i][1]])
                top_scores = top_scores.reshape((1,2))
                continue

            dist_i = np.min(np.abs(top_scores[:, 0] - ref_ind))
            if dist_i >= pat_len / 5:
                top_scores = np.vstack((top_scores, [ref_ind, corr_scores[i][1]]))

            if len(top_scores) == 9:
                break
        top_scores = np.array(top_scores)
        self.top_scores = top_scores

        return top_scores


class Reddit:


    def __init__(self, client_id, client_secret, user_agent, subreddit_name):

        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.subreddit_name = subreddit_name
        self.subreddit = praw.Reddit(client_id=self.client_id,
                                     client_secret=self.client_secret,
                                     user_agent=self.user_agent).subreddit(
                                         self.subreddit_name)


    # type(subreddit) = praw.models.reddit.subreddit.Subreddit
    # sort_by: 'hot', 'new', 'top', 'rising'
    # If search != None and type(search) = str then that will override sort_by
    def submission_getter(
        self,
        sort_by='top',
        search=None,
        search_sort_by='relevance',
        no_of_submissions=10):

        print('starting submission getter')

        if isinstance(search, str):
            submissions = self.subreddit.search(search, sort=search_sort_by)

        elif sort_by == 'top':
            submissions = self.subreddit.top(limit=no_of_submissions)
        elif sort_by == 'hot':
            submissions = self.subreddit.hot(limit=no_of_submissions)
        elif sort_by == 'new':
            submissions = self.subreddit.new(limit=no_of_submissions)
        elif sort_by == 'rising':
            submissions = self.subreddit.rising(limit=no_of_submissions)

        submission_list = []
        count = 1
        for sub in submissions:
            submission_list.append(sub)
            if count == no_of_submissions:
                break
            count += 1

        self.submission_list = submission_list

        return submission_list


    def submissions_to_comments_dic(self, no_of_comments=10):

        print('starting submissions to comments dic')

        submission_comments = {submission : [] for submission in
                              self.submission_list}
        
        for submission in submission_comments.keys():
            submission_comments[submission] = submission.comments[: no_of_comments]

        self.submission_comments = submission_comments

        return submission_comments


    def comment_replies(self, no_of_replies=10):

        print('starting comments replies')

        submissions_comments_replies = {subs : {} for subs in
                                        self.submission_list}

        for subs in self.submission_list:
            comments_replies = {coms : [] for coms in
                               self.submission_comments[subs]}
            count_c = 1
            for coms in self.submission_comments[subs]:
                print(f'COMMENT {count_c}')
                replies = coms.replies
                replies.replace_more(limit=None)
                replies = replies[: no_of_replies]
                for reply in  replies:
                    comments_replies[coms].append(reply)
                count_c += 1

            submissions_comments_replies[subs] = comments_replies

        self.submissions_comments_replies = submissions_comments_replies

        return submissions_comments_replies


if __name__ == '__main__':


    wsb = Reddit(client_id=client_id,
                 client_secret=client_secret,
                 user_agent=user_agent,
                 subreddit_name='wallstreetbets')

    wsb.submission_getter(search='Discussion', no_of_submissions=1)
    wsb.submissions_to_comments_dic(no_of_comments=3)
    subs_coms_reps = wsb.comment_replies()

    for subs in subs_coms_reps.keys():
        print('--------------------')
        print(f'subs.title\n')
        print('--------------------\n')
        for coms in subs_coms_reps[subs].keys():
            print(f'Comment:\n{coms.body}\n')
            for reps in subs_coms_reps[subs][coms]:
                print(f'Reply:\n{reps.body}')




                 

    exit()

    print('-----Testing-----')
    data_path='/Users/nickeisenberg/GitRepos/Python_Misc/Notebook/DataSets/gme_11_3_22.csv'
    df = pd.read_csv(data_path)
    df.rename(columns={df.columns[0] : 'Date'}, inplace=True)

    dfhead = df.iloc[:10]

    e_date ='2020-11-05 09:30:00-05:00'
    gme_vol = Volitility(data=df)
    gme_vol_open = gme_vol.historical(e_date=e_date, method='open')

    gme_vol_head = Volitility(data=dfhead)
    gme_vol__head_open = gme_vol_head.historical(method='open')

    print(gme_vol.volitility)
    print(gme_vol.mean_return)
    print(gme_vol.dt)
    print('----------------')
    print(gme_vol_head.volitility)
    print(gme_vol_head.mean_return)
    print(gme_vol_head.dt)



