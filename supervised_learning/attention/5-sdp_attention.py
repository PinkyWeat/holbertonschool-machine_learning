#!/usr/bin/env python3
"""the RNN Decoder Module"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ calculates the scaled dot product attention """
    scores = tf.matmul(Q, K, transpose_b=True) / \
             tf.sqrt(tf.cast(Q.shape[-1], 'float32'))

    if mask:
        scores = mask * -1e9

    weights = tf.nn.softmax(scores)
    output = tf.matmul(weights, V)

    return output, weights
