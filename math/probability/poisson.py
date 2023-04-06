#!/usr/bin/env python3
"""Poisson"""


class Poisson():
    """Poisson Distribution"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)) / len(data)

    def pmf(self, k):
        """calculates the value of the PMF for a given number of “successes”"""
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)

        # calculates
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n - 1)

        e = 2.7182818285
        return ((self.lambtha ** k) * (1 / (e ** self.lambtha))) / factorial(k)
