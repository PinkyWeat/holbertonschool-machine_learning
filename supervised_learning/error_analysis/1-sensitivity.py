#!/usr/bin/env python3
"""  """
import numpy as np


def sensitivity(confusion):
    """ calculates sensitivity for each class in confusion matrix """

    id_matrix = np.identity(confusion.shape[1])
    TP = np.sum(confusion * id_matrix, axis=1)
    P = np.sum(confusion, axis=1)
    return TP / P
