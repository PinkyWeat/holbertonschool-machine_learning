#!/usr/bin/env python3
"""Matrix Transpose"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""

    # first for column to become rows
    new_matrix = [[matrix[j][i] for j in range(len(matrix))]
                  for i in range(len(matrix[0]))]  # rows to become columns
    return new_matrix
