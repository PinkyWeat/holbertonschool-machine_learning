#!/usr/bin/env python3
""" Pandas - Flip it, Switch it """

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.sort_index(ascending=False)

df = df.transpose()

print(df.tail(8))
