#!/usr/bin/env python3
""" Pandas - From Dictionary """
import pandas as pd


data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

df = pd.DataFrame(data)

df.index = ['A', 'B', 'C', 'D']
