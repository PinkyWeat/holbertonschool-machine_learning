#!/usr/bin/env python3
""" Preprocessed data for BTC """
preprocess_data = __import__('preprocess_data').preprocess_data
implement_dataset = __import__("forecast_btc").implement_dataset


def main():
    csv_path = "../data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    preprocess_data(csv_path)
    implement_dataset()


if __name__ == '__main__':
    main()
