#!/usr/bin/env python3
"""  Error Analysis """
import numpy as np


def specificity(confusion):
    """ calculates specificity for each class in the confused matrix """
    TP = np.diag(confusion)

    # False Positives
    FP = np.sum(confusion, axis=0) - TP

    # True Negatives
    total = np.sum(confusion)
    TN = total - np.sum(confusion, axis=1) - np.sum(confusion, axis=0) + TP

    specificity = TN / (TN + FP)

    return specificity
