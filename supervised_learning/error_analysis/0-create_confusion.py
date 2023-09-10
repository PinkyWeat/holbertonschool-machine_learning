#!/usr/bin/env python3
""" Error Analysis """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix """
    return np.dot(labels.T, logits)
