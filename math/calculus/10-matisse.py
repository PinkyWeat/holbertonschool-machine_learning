#!/usr/bin/env python3
"""math to code"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    # check type
    if not isinstance(poly, list):
        return None

    # if less than 2 terms - derivative == 0
    if len(poly) < 2:
        return [0]

    # derivative * each coefficient by its power
    derivative = [poly[i] * i for i in range(1, len(poly))]

    # If the derivative is 0, return [0]
    if all(c == 0 for c in derivative):
        return [0]

    return derivative
