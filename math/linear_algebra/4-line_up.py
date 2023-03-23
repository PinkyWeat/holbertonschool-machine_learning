#!/usr/bin/env python3
"""Line Up"""


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    arr3 = []
    if len(arr1) == len(arr2):
        arr3 = arr1 + arr2
    else:
        return None

    return arr3
