#!/usr/bin/env python3
""" Pandas - From File """
import pandas as pd


def from_file(filename, delimiter):
    """ Loads data from a file as a pd.DataFrame """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
