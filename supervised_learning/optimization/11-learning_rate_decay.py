#!/usr/bin/env python3
""" Optimization """


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ updates learning rate using inverse time decay in np """

    alpha_new = alpha / (1 + decay_rate * (global_step // decay_step))
    return alpha_new
