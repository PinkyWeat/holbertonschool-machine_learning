#!/usr/bin/env python3
"""Self Attention Module"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ calculate the attention for machine translation based on a paper"""

    def __init__(self, units):
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, s_prev, hidden_states):
        """s_prev is a tensor of shape (batch, units) containing the previous
        decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder

        Returns: context, weights
            context is a tensor of shape (batch, units) that contains the
            context vector for the decoder
            weights is a tensor of shape (batch, input_seq_len, 1) that
            contains the attention weights"""
        query_with_time_axis = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(query_with_time_axis) +
                                  self.U(hidden_states)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context = attention_weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights
