#!/usr/bin/env python3
"""Task 4 - Loss"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""

    # cross-entropy loss for each instance in batch
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y)

    # mean loss over entire batch
    mean_loss = tf.reduce_mean(loss)
    return mean_loss
