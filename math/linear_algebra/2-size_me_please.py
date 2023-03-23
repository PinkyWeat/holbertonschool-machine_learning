#!/usr/bin/env python3
"""Shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a given matrix"""
    shape = []

    while type(matrix) != int:
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape
