#!/usr/bin/env python3
"""Ridin Bareback"""


def mat_mul(mat1, mat2):
    """multiply matrix !!! """

    n_rows_mat1 = len(mat1)
    n_cols_mat1 = len(mat1[0])
    n_rows_mat2 = len(mat2)
    n_cols_mat2 = len(mat2[0])

    if n_cols_mat1 != n_rows_mat2:
        return

    # in order to re-assign values
    new_matrix = [[0 for j in range(n_cols_mat2)] for i in range(n_rows_mat1)]

    try:
        for rows_mat1 in range(n_rows_mat1):
            for cols_mat2 in range(n_cols_mat2):
                for cols_mat1 in range(n_cols_mat1):
                    new_matrix[rows_mat1][cols_mat2] += \
                        mat1[rows_mat1][cols_mat1] * mat2[cols_mat1][cols_mat2]
    except ValueError:
        return
    return new_matrix
