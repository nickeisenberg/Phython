import numpy as np
from sklearn.preprocessing import MinMaxScaler

# turn off scientific notation
np.set_printoptions(suppress=True)


#-statistics-functions---------------------------------------------------------

# need to remove the gmt offset that comes with yfinance date/time outputs
def historical_volitility(stock_price,
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

    return top_scores
