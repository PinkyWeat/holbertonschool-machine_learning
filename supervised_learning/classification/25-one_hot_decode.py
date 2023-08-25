#!/usr/bin/env python3
""" Tasks: One-Hot Matrix """
import numpy as np


import numpy as np


def decode_one_hot(one_hot):
    """ Decodes a one-hot matrix into a label vector """
    try:
        classes = np.arange(one_hot.shape[0])
        decoded_matrix = np.dot(classes, one_hot)
        return decoded_matrix.astype(int)
    except Exception as e:
        return None
