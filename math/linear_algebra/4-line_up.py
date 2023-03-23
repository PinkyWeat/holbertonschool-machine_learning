#!/usr/bin/env python3
"""Line Up"""


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    arr3 = []
    if len(arr1) == len(arr2):
        for elem in range(len(arr1)):
            arr3.append(arr1[elem] + arr2[elem])
    else:
        return None

    return arr3
