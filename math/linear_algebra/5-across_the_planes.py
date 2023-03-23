#!/usr/bin/env python3
"""Across The Planes"""


def add_matrices2D(mat1, mat2):
    """adds two matrix element-wise"""

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    sum = []

    for rows in range(len(mat1)):
        sum_row = []
        for columns in range(len(mat1[0])):
            sum_row.append(mat1[rows][columns] + mat2[rows][columns])
        sum.append(sum_row)

    return sum
