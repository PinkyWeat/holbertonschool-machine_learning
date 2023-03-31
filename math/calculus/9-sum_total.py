#!/usr/bin/env python3
"""math to code"""


def summation_i_squared(n):
    """calculates sum_{i=1}^{n} i^2"""
    if not isinstance(n, int) and n < -1:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
