#!/usr/bin/env python3
def matrix_shape(matrix):
    shape = []
    if isinstance(matrix[0], list) and type(int):
        slices = len(matrix)
        rows = len(matrix[0])
        if isinstance(matrix[0][0], list) and type(int):
            columns = len(matrix[0][0])
            shape = [slices, rows, columns]
        shape = [slices, rows]
    return shape
