#!/usr/bin/env python3
""" Optimization """
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1,
                   beta2, epsilon):
    """ creates training op for neural netwk in tflow using
    the Adam optimization algorithm"""
    return tf.train.AdamOptimizer(alpha,
                                  beta1=beta1, beta2=beta2,
                                  epsilon=epsilon).minimize(loss)
