#!/usr/bin/env python3
""" Preprocess and split data for BTC """
import pandas as pd


def preprocess_data(csv_path):
    """ load and preprocess """
    wip = pd.read_csv(csv_path)
    wip['Timestamp'] = pd.to_datetime(wip['Timestamp'], unit='s')
    wip.sort_values(by='Timestamp', inplace=True)
    wip = wip[wip['Timestamp'] >= pd.to_datetime("2017-01-01")]
    wip.set_index('Timestamp', inplace=True)
    wip = wip.drop(['High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)'], axis=1)
    wip = wip.resample('h').mean()
    wip.dropna(inplace=True)

    mean = wip.mean()
    std = wip.std()
    wip_normalized = (wip - mean) / std

    # Split data into train, validation, and test sets
    n = len(wip_normalized)
    train_df = wip_normalized[:int(n * 0.7)]
    val_df = wip_normalized[int(n * 0.7):int(n * 0.9)]
    test_df = wip_normalized[int(n * 0.9):]

    # Differencing the data to make it more stationary
    train_df = train_df.diff().dropna()
    val_df = val_df.diff().dropna()
    test_df = test_df.diff().dropna()

    wip_normalized.to_csv('normalized_data.csv')
    train_df.to_csv('train_data.csv')
    val_df.to_csv('val_data.csv')
    test_df.to_csv('test_data.csv')

    print("Data processed and split into train, validation, and test sets.")
    return train_df, val_df, test_df
