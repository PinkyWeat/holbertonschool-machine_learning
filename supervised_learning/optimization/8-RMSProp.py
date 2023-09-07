#!/usr/bin/env python3
""" Optimization """
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates training op for neural netwk in tf
    using RMSProp op algorithm """

    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)

    return train_op
