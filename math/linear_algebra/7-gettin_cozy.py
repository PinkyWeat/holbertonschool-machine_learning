#!/usr/bin/env python3
"""Gettin Cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""

    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2

    if axis == 1 and len(mat1) == len(mat2):
        return [mat1[i] + mat2[i] for i in range(len(mat2))]

    return
