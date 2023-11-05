#!/usr/bin/env python3
""" Advanced Linear Algebra """


def determinant(matrix):
    """ calculates the determinant of a matrix """
    if matrix == [[]]:
        return 1

    if not matrix or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not matrix[0] or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(len(matrix[0])):
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        minor_det = determinant(minor)
        cofactor = (-1) ** col * minor_det
        det += matrix[0][col] * cofactor

    return det
