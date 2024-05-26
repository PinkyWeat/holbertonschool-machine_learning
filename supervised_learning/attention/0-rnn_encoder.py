#!/usr/bin/env python3
"""RNN Encoder Module"""
import tensorflow as tf


class RNNEncoder:
    """ RNN Encoder """

    def __init__(self, vocab, embedding, units, batch):
        """ the init """
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """ the initialize hidden state """
        return tf.zeros((self.batch, self.units))

    def __call__(self, x, initial):
        """ the call """
        embedded = self.embedding(x)

        return self.gru(embedded, initial_state=initial)
