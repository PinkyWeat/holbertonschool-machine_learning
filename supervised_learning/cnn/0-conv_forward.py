#!/usr/bin/env python3
""" Convolutional Neural Networks """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs frwd prop over convolutional layer of a neural netwk """
    num_samples, prev_height, prev_width, prev_channels = A_prev.shape
    filter_height, filter_width, _, num_filters = W.shape
    stride_height, stride_width = stride

    if padding == "valid":
        padding_height, padding_width = 0, 0
    elif padding == "same":
        padding_height = int((((prev_height - 1) * stride_height) +
                              filter_height - prev_height) / 2)
        padding_width = int((((prev_width - 1) * stride_width) +
                             filter_width - prev_width) / 2)
    else:
        raise ValueError("Padding type not recognized")

    padded_A_prev = np.pad(A_prev, ((0, 0), (padding_height, padding_height),
                                    (padding_width, padding_width), (0, 0)),
                           "constant", constant_values=0)

    output_height = int((prev_height + 2 * padding_height -
                         filter_height) / stride_height + 1)
    output_width = int((prev_width + 2 * padding_width -
                        filter_width) / stride_width + 1)

    conv_output = np.zeros((num_samples, output_height,
                            output_width, num_filters))

    for h in range(output_height):
        vertical_offset = h * stride_height
        for w in range(output_width):
            horizontal_offset = w * stride_width

            slice_A_prev = padded_A_prev[:, vertical_offset:
                                         vertical_offset + filter_height,
                                         horizontal_offset:
                                         horizontal_offset + filter_width, :]

            for f in range(num_filters):
                conv_output[:, h, w, f] = np.sum(slice_A_prev * W[:, :, :, f],
                                                 axis=(1, 2, 3))

    return activation(output_img + b)
