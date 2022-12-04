# Data:
# 12 currency pairs (time series) over roughly 5 years

from common import load_data_structure
import math, datetime
from collections import deque

def time_features(dt):
    min_f = math.sin(2 * math.pi * dt.minute / 60.)
    hour_f = math.sin(2 * math.pi * dt.hour / 24.)
    day_f = math.sin(2 * math.pi * dt.weekday() / 7.)
    return min_f, hour_f, day_f

def parse_time(text):
    year = int(text[0:4])
    month = int(text[5:7])
    day = int(text[8:10])

    hour = int(text[11:13])
    mins = int(text[14:16])
    sec = int(text[17:19])
    return datetime.datetime(year, month, day, hour, mins, sec)

class Data():
    def __init__(self, closing_path, states_path, T):
        self.closing = None
        self.state_space = None
        self.closing_path = closing_path
        self.states_path = states_path
        self.T = T
        self.n = 0

        self.load()

    def load(self):
        # to do list: build the json file for each stock
        self.closing = load_data_structure(self.closing_path)
        self.state_space = load_data_structure(self.states_path)
        # print(len(self.state_space))
        self.it = self.iterator()

    def next(self):
        self.n += 1
        return next(self.it)

    def iterator(self):
        d = deque()
        for v in zip(self.closing[7:], self.state_space):
        # for v in zip(self.closing[7:], self.state_space):
            closing = v[0][1]
            time = time_features(parse_time(v[0][0]))
            features = v[1][1]
            features_total = features
            features_total.extend(time)

            d.append(features_total)
            while len(d) > self.T:
                d.popleft()

            if len(d) == self.T:
                yield closing, list(d)

    def reset(self):
        self.it = self.iterator()
        self.n = 0


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from datetime import timedelta
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean
        

def time_features2(dates, freq='h'):
    dates['month'] = dates.date.apply(lambda row:row.month,1)
    dates['day'] = dates.date.apply(lambda row:row.day,1)
    dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
    dates['hour'] = dates.date.apply(lambda row:row.hour,1)
    dates['minute'] = dates.date.apply(lambda row:row.minute,1)
    dates['minute'] = dates.minute.map(lambda x:x//15)
    freq_map = {
        'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
        'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
        't':['month','day','weekday','hour','minute'],
    }
    return dates[freq_map[freq.lower()]].values

class Dataset_Pred(Dataset):
    def __init__(self, dataframe, size=None, scale=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.dataframe = dataframe
        
        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = self.dataframe
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        delta = df_raw["date"].iloc[1] - df_raw["date"].iloc[0]
        if delta>=timedelta(hours=1):
            self.freq='h'
        else:
            self.freq='t'

        

        border1 = 0
        border2 = len(df_raw)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]


        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1