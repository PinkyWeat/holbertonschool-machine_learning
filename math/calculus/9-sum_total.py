#!/usr/bin/env python3
"""math to code"""


def summation_i_squared(n):
    """calculates sum_{i=1}^{n} i^2"""
    if type(n) != int and n < 0:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
