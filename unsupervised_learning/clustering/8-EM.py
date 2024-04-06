#!/usr/bin/env python3
""" Clustering """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ performs the expectation maximization for a GMM """
    if (not isinstance(verbose, bool) or not isinstance(tol, float) or
            not isinstance(iterations, int) or iterations < 1):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    prev_l = []

    for i in range(iterations + 1):
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, likelihood = expectation(X, pi, m, S)
        if i == iterations:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i, round(likelihood, 5)))
            break

        if len(prev_l) and abs(likelihood - prev_l[-1]) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i, round(likelihood, 5)))
            break

        prev_l.append(likelihood)
        pi, m, S = maximization(X, g)

        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(likelihood, 5)))

    return pi, m, S, g, likelihood
