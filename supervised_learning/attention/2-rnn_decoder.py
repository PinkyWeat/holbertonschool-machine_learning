#!/usr/bin/env python3
"""the RNN Decoder Module"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ the rnn decoder """

    def __init__(self, vocab, embedding, units, batch):
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def __call__(self, x, s_prev, hidden_states):
        """ the call """
        attention = SelfAttention(self.units)
        context, _ = attention(s_prev, hidden_states)

        x = self.embedding(x)

        context_x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        outputs, s = self.gru(context_x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))

        y = self.F(outputs)

        return y, s
