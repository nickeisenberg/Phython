from sys import exit
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import scipy.signal as sps
from sklearn.preprocessing import MinMaxScaler

gme_2y = pd.read_csv('/Users/nickeisenberg/GitRepos/Phython/DataSets/gme_11_3_22.csv')
cols = gme_2y.columns
gme_2y.rename(columns={cols[0] : 'Date'}, inplace=True)
dates = gme_2y['Date'].values

gme_2y = gme_2y['Open'].values.reshape((-1,1))
Mm = MinMaxScaler(feature_range=(0,1))
gme_scaled = Mm.fit_transform(gme_2y)

time = np.linspace(0, 1, len(gme_2y)).reshape((-1,1))

low = int(.9182 * len(time))
up = int(.9835 * len(time))

# print((len(time) - up) / 7.5)
# print((up - low) / 7.5)
# 
# plt.plot(time, gme_scaled)
# plt.show()

gme_pat = gme_scaled[low:up]
pat_len = len(gme_pat)

def pearson_corr(x, y):
    ux, uy = np.mean(x), np.mean(y)
    x_cen, y_cen = x - ux, y - uy
    return np.dot(x_cen, y_cen) / (np.sqrt(np.sum(np.square(x_cen))) * np.sqrt(np.sum(np.square(y_cen))))

# Pearson correlation
corr_scores = []
for i in range(0, low + 1):
    gme_ref = gme_scaled[i : i + pat_len]
    score = pearson_corr(gme_pat.reshape(-1), gme_ref.reshape(-1))
    corr_scores.append([i, score])
corr_scores = np.array(corr_scores)
corr_scores = corr_scores[corr_scores[:, 1].argsort()][::-1]

#-Normalized-dot-product-for-the-def-of-corr--
# gme_pat_norm = gme_pat / np.sqrt(np.sum(np.multiply(gme_pat, gme_pat)))
# corr_scores = []
# for i in range(0, low + 1):
#     gme_ref = gme_scaled[i : i + pat_len]
#     gme_ref_norm = gme_ref / np.sqrt(np.sum(np.multiply(gme_ref, gme_ref)))
#     score = np.sqrt(np.sum(np.multiply(gme_ref_norm, gme_pat_norm)))
#     corr_scores.append([i, score])
# corr_scores = np.array(corr_scores)
# corr_scores = corr_scores[corr_scores[:, 1].argsort()][::-1]

top_scores = [] # index for the start of the correlation reference
for i in range(len(corr_scores[:, 0])):
    val_ind = corr_scores[i][0]
    if np.abs(val_ind -low) < pat_len:
        continue

    if len(top_scores) == 0:
        top_scores.append([val_ind, corr_scores[i][1]])
        continue

    dists_i = []
    for ts in top_scores:
        dists_i.append(np.abs(ts[0] - val_ind))
    dists_i = np.array(dists_i)
    if np.min(dists_i) >= pat_len / 5: # chose how close each start point can be
        top_scores.append([val_ind, corr_scores[i][1]])
    if len(top_scores) == 9:
        break
top_scores = np.array(top_scores)

print(top_scores)

for i in range(8):
    ind = int(top_scores[i][0])
    corr_past = gme_2y[ind : ind + pat_len]
    time_past = time[ind : ind + pat_len]
    corr_fut = gme_2y[ind + pat_len : ind + 3 * pat_len]
    time_fut = time[ind + pat_len : ind + 3 * pat_len]
    plt.subplot(3, 3, i+1)
    plt.title(f'Start : {dates[ind]}\nEnd : {dates[ind + 3 * pat_len]}\n Blue to Green Corr : {top_scores[i][1]}')
    plt.plot(time_past, corr_past, c='blue')
    plt.plot(time_fut, corr_fut, c='darkorange')

plt.subplot(339)
plt.title(f'Start : {dates[low]}\nEnd : {dates[-1]}\nGreen : Pattern for correlation')
plt.plot(time[low:up], gme_2y[low:up], c='green')
plt.plot(time[up:], gme_2y[up:], c='darkorange')

plt.tight_layout()
plt.suptitle('Movement of GME following correlated price movements')
plt.show()
