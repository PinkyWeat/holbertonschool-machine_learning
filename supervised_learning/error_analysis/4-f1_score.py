#!/usr/bin/env python3
"""  Error Analysis """
import numpy as np


def f1_score(confusion):
    """ calculates F1 score of a confusion matrix """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    # Calculating sensitivity && precision
    recall = sensitivity(confusion)
    prec = precision(confusion)

    # F1 score
    f1 = 2 * (prec * recall) / (prec + recall)

    return f1
