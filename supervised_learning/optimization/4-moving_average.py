#!/usr/bin/env python3
""" Optimization """
import numpy as np


def moving_average(data, beta):
    """ Calculates weighted moving average """

    weights = 0
    wma = []

    for i in range(len(data)):
        weights = beta * weights + (1 - beta) * data[i]
        bias_correction = 1 - (beta ** (i + 1))
        wma.append(weights / bias_correction)
    return wma
