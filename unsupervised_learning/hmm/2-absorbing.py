#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np
# P is a square 2D numpy.ndarray of shape (n, n)
# representing the standard transition matrix
# P[i, j] is the probability of transitioning from state i to state j
# n is the number of states in the markov chain
# Returns: True if it is absorbing, or False on failure


def absorbing(P):
    """ determines if a markov chain is absorbing """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False

    n = P.shape[0]
    absorbing_states = [i for i in range(n) if P[i, i] == 1
                        and np.all(P[i, :] == np.eye(n)[i, :])]

    if not absorbing_states:
        return False

    for i in range(n):
        if i not in absorbing_states:
            can_reach = False
            for power in range(1, n + 1):
                P_power = np.linalg.matrix_power(P, power)
                for absorbing_state in absorbing_states:
                    if P_power[i, absorbing_state] > 0:
                        can_reach = True
                        break
                if can_reach:
                    break
            if not can_reach:
                return False

    return True
