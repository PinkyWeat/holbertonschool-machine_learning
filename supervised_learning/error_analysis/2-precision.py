#!/usr/bin/env python3
"""  Error Analysis """
import numpy as np


def precision(confusion):
    """ calculates precision for each class in confusion matrix """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP

    precision = TP / (TP + FP)

    return precision
