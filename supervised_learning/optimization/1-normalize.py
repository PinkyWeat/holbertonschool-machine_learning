#!/usr/bin/env python3
""" Optimization """
import numpy as np


def normalize(X, m, s):
    """ standarizes a matrix"""
    return ((X - m) / s)
