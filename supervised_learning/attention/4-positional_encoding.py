#!/usr/bin/env python3
"""the RNN Decoder Module"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """ calculates the positional encoding for a transformer """
    PE = np.zeros((max_seq_len, dm))
    for k in range(max_seq_len):
        for i in np.arange(int(dm / 2)):
            n_pow = np.math.pow(10000, 2 * i / dm)
            PE[k, 2 * i] = np.math.sin(k / n_pow)
            PE[k, 2 * i + 1] = np.math.cos(k / n_pow)

    return PE
