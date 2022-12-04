from collections import OrderedDict, deque
import numpy as np
import json, codecs
from data_process_common import *
import os
from common import load_data_structure
import math, datetime
from collections import deque

import pandas as pd
df = pd.read_csv('../../project_data/m10_upbit_min_data_KRW-ADA_20210320_20220324.csv')

time_list = np.array(df['candle_date_time_utc'])
open_price_list = np.array(df['opening_price'])
close_price_list = np.array(df['trade_price'])

buckets = OrderedDict()
for t, o_p, c_p in zip(time_list, open_price_list, close_price_list):
    t= parse_time(t)
    dt0 = t
    printed_time = str(dt0.replace(minute=((t.minute // 30)) * 30, second=0))

    if printed_time not in buckets:
        buckets[printed_time] = []

    buckets[printed_time].append((t, o_p, c_p))

# print(list(buckets.items())[1])

# calculate ohlc data
ohlc = OrderedDict()
for t, bucket in buckets.items():
    ohlc[t] = get_ohlc(bucket)

# print(list(ohlc.items())[1])

closing = list(map(lambda t_v: (t_v[0], t_v[1][3][2]), ohlc.items()))
# print(closing[0:5])

# calculate 8-delayed-log-returns of closing prices
n = 8
log_returns = []
lag = deque()
last_price = None
for t, v in closing:
    if last_price is not None:
        lag.append(math.log(v / last_price))
        while len(lag) > n:
            lag.popleft()

        if len(lag) == n:
            log_returns.append((t, list(lag)))
    last_price = v

# print(log_returns[0:5])

# z-score normalization, group 96 states into a cluster
z_score_clusters = OrderedDict()
for n, t_vs in enumerate(log_returns):
    i = n // 84
    if i not in z_score_clusters:
        z_score_clusters[i] = []
    z_score_clusters[i].append(t_vs[1])

z_score_transformed = []
for n, t_vs in enumerate(log_returns):
    i = n // 84
    mean, variance = calc_z_scores_parameters(z_score_clusters[i])
    z_score_transformed.append([t_vs[0], z_transform(t_vs[1], mean, variance)])


data_name = "KRW-ADA_20210320_20220324_84"
save_data_structure(z_score_transformed, "./data_states/" + data_name + "-states.json")
save_data_structure(closing, "./data_closing/" + data_name + "-closing.json")
print('save file {}'.format(data_name))
