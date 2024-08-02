#!/usr/bin/env python3
""" Pandas - From Numpy """
import numpy as np
import pandas as pd

def from_numpy(array):
    """ creates a pd.DataFrame from a np.ndarray """
    num_columns = array.shape[1]  # determines n of columns in array

    # generates list of clmn names starting A onwards in mayus
    column_names = [chr(65 + i ) for i in range(num_columns)]

    data_frame = pd.DataFrame(array, columns=column_names)

    return data_frame
