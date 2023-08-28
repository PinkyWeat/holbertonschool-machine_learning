#!/usr/bin/env python3
"""Task 5 - Create Training Op"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """ Creates the training operation for the networks """
    # Defines Gradient Descent optimizer w/ given learning rate
    optimizer = tf.train.GradientDescentOptimizer(alpha)

    # training op minimizes loss
    train_op = optimizer.minimize(loss)

    return train_op
