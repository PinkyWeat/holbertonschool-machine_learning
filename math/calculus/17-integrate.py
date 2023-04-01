#!/usr/bin/env python3
"""math to code"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    # border cases
    if type(poly) != list or type(C) != int:
        return None
    if len(poly) == 0:
        return None

    integral = [C]
    if poly == [0]:
        return integral
    idx = 1
    for i in poly:
        integral.append(i / idx)
        if integral[idx] % 1 == 0:
            integral[idx] = int(integral[idx])
        idx += 1

    return integral
