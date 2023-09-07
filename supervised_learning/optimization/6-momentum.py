#!/usr/bin/env python3
""" Optimization """
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """ Creates training op w/ gradient descent w/ momentum op algorithm """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)

    return train_op
