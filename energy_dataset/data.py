from random import shuffle
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd
import glob
import numpy as np

class BaseDataset:
    def __init__(
        self,
        root,
        train_split=0.2,
        test_split=0.2,
        lookback=50
    ):
        self.root = root
        self.train_split = train_split
        self.test_split = test_split
        self.lookback = lookback
        self.label_scalers = {}
        self.train_x = []
        self.train_y = []
        self.val_x = []
        self.val_y = []
        self.test_x = {}
        self.test_y = {}

        self.process_and_combine_csvs()
    
    def process_and_combine_csvs(self):
        csv_files = glob.glob(self.root + '*.csv')

        print("CSV Files: " + str(csv_files))

        for file in csv_files:
            if file == self.root + "pjm_hourly_est.csv":
                continue
            print("Processing File: " + file)

            df = pd.read_csv(file, parse_dates=[0])
            df = self.convert_to_datetime(df)
            norm_data = self.normalise_data(df, file)

            inputs = np.zeros((len(norm_data)-self.lookback,self.lookback,df.shape[1]))
            labels = np.zeros(len(norm_data)-self.lookback)

            for i in range(self.lookback, len(norm_data)):
                inputs[i-self.lookback] = norm_data[i-self.lookback:i]
                labels[i-self.lookback] = norm_data[i,0]

            inputs = inputs.reshape(-1,self.lookback,df.shape[1])
            labels = labels.reshape(-1,1)

            total_hold_out = self.train_split + self.test_split
            total_hold_portion = int((total_hold_out) * len(inputs))
            val_portion = int((self.train_split) * len(inputs))

            if len(self.train_x) == 0:
                self.train_x = inputs[:-total_hold_portion]
                self.train_y = labels[:-total_hold_portion]
                self.val_x = inputs[-total_hold_portion:-val_portion]
                self.val_y = labels[-total_hold_portion:-val_portion]
            else:
                self.train_x = np.concatenate((self.train_x, inputs[:-total_hold_portion]))
                self.train_y = np.concatenate((self.train_y, labels[:-total_hold_portion]))
                self.val_x = np.concatenate((self.val_x, inputs[-total_hold_portion:-val_portion]))
                self.val_y = np.concatenate((self.val_y, labels[-total_hold_portion:-val_portion]))

            self.test_x[file] = (inputs[-val_portion:])
            self.test_y[file] = (labels[-val_portion:])

    def convert_to_datetime(self, df):
        df["hour"] = df.apply(lambda x: x['Datetime'].hour,axis=1)
        df["dayofweek"] = df.apply(lambda x: x['Datetime'].dayofweek,axis=1)
        df['month'] = df.apply(lambda x: x['Datetime'].month,axis=1)
        df['dayofyear'] = df.apply(lambda x: x['Datetime'].dayofyear,axis=1)
        df = df.sort_values("Datetime").drop("Datetime",axis=1)
        return df


    def normalise_data(self, df, file):
        scale = MinMaxScaler()
        label_scale = MinMaxScaler()
        norm_data = scale.fit_transform(df.values)
        label_scale.fit(df.iloc[:,0].values.reshape(-1,1))
        self.label_scalers[file] = label_scale
        return norm_data