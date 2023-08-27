#!/usr/bin/env python3
"""Task 2 - Forward Propagation"""

import tensorflow.compat.v1 as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""

    prev_layer_output = x

    for i, (size, activation) in enumerate(zip(layer_sizes, activations)):
        # Create a layer and use its output as input for the next layer
        prev_layer_output = create_layer(prev_layer_output, size, activation)

    return prev_layer_output
