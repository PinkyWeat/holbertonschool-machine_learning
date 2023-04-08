#!/usr/bin/env python3
"""Normal Distribution"""


class Normal():
    """represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            s = sum(map(lambda x: (x - self.mean) ** 2, data))
            self.stddev = (s / len(data)) ** 0.5
