#!/usr/bin/env python3
"""Task 3 - Prediction accuracy"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""

    # tf.equal(predicted class, true class) - returns boolean tensor
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

    # mean of tensor = decimal accuracy of prediction
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
