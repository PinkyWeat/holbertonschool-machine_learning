#!/usr/bin/env python3
"""Bayesian Information Criterion Module"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ Finds the best number of clusters for a GMM using the Bayesian
        source
     """

    try:
        if kmax == 1:
            return None, None, None, None
        n, d = X.shape
        if kmax is None:
            kmax = n
            if kmax >= kmin:
                return None, None, None, None

        k_history = list(range(kmin, kmax+1))
        results_history = []
        lh_history = []
        bic_history = []

        for k in range(kmin, kmax+1):
            pi, m, S, g, lh = expectation_maximization(X, k, iterations, tol,
                                                       verbose)

            if pi is None or m is None or S is None or g is None or lh is None:
                return None, None, None, None

            num_parameters = k + k * d + k * d * (d + 1) // 2 - 1
            bic = num_parameters * np.log(n) - 2 * lh

            lh_history.append(lh)
            results_history.append((pi, m, S))
            bic_history.append(bic)

        min_bic_index = np.argmin(bic_history)
        best_k = k_history[min_bic_index]
        best_result = results_history[min_bic_index]

        return best_k, best_result, np.array(lh_history), np.array(bic_history)
    except Exception:
        return None, None, None, None