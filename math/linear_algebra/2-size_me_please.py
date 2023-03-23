#!/usr/bin/env python3
def matrix_shape(matrix):
    slices = len(matrix)
    rows = len(matrix[0])
    if isinstance(matrix[0][0], list):
        columns = len(matrix[0][0])
        return [slices, rows, columns]
    return [slices, rows]
