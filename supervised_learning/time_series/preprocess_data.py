#!/usr/bin/env python3
""" Preprocessed data for BTC """
import pandas as pd


def preprocess_data(csv_path):
    """ preprocesses raw data """
    wip = (pd.read_csv(csv_path)
           .assign(Timestamp=lambda x:
                   pd.to_datetime(x['Timestamp'], unit='s'))
           .sort_values(by='Timestamp')
           .loc[lambda x: x['Timestamp'] >= pd.to_datetime("2017-01-01")]
           .set_index('Timestamp')
           .drop(['High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)'], axis=1)
           .resample('H').mean()
           .dropna())

    # Normalize data
    mean = wip.mean()
    std = wip.std()
    wip = (wip - mean) / std

    wip.to_csv('preprocessed_data.csv')
    print(wip)
    return wip
