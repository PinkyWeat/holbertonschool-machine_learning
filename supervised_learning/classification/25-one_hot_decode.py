#!/usr/bin/env python3
""" Tasks: One-Hot Matrix """
import numpy as np


def one_hot_decode(one_hot):
    """ Converts a numeric label vector into a one-hot matrix """
    try:
        decoded_matrix = np.zeros(shape=(one_hot.shape[1]))
        classes = np.array(range(one_hot.shape[0]))

        pre_decode = (one_hot.T * classes).T
        for i in pre_decode:
            decoded_matrix += i
        return decoded_matrix.astype(int)
    except Exception as e:
        return None
