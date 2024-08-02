#!/usr/bin/env python3
""" Pandas - Visualize Data """
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])

df = df.rename(columns={'Timestamp': 'Date'})

df['Date'] = pd.to_datetime(df['Date'], unit='s')

df = df.set_index('Date')

df['Close'] = df['Close'].fillna(method='ffill')

df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

df = df[df.index >= '2017-01-01']

df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
plt.figure(figsize=(14, 7))

plt.plot(df_daily.index, df_daily['Open'], label='Open')
plt.plot(df_daily.index, df_daily['High'], label='High')
plt.plot(df_daily.index, df_daily['Low'], label='Low')
plt.plot(df_daily.index, df_daily['Close'], label='Close')
plt.plot(df_daily.index, df_daily['Volume_(BTC)'], label='Volume (BTC)')
plt.plot(df_daily.index, df_daily['Volume_(Currency)'], label='Volume (Currency)')

plt.title('Daily Cryptocurrency Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()