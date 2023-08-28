#!/usr/bin/env python3
"""Task 7 - Evaluate"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """evaluating"""
    with tf.Session() as sess:
        # Load the saved model
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        # Retrieve the necessary tensors from the loaded graph
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        # Evaluate tensors
        prediction, acc, cost = sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y})

    return prediction, acc, cost
