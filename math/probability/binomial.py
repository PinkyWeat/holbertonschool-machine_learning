#!/usr/bin/env python3
"""Probability Distribution"""


class Binomial():
    """represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)
            if 0 > p or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")

            # find avg
            mean = sum(data) / len(data)
            # find measure of dispersion
            variance = sum((x - mean)**2 for x in data) / (len(data))

            # calc n first
            self.n = round(mean / (1 - (variance / mean)))
            self.p = float(mean / self.n)
