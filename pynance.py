import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

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
                                             prepost=prepost,
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

if __name__ == '__main__':

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



