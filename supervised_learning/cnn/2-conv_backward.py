#!/usr/bin/env python3
""" Convolutional Neural Network """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ performs backpropagation over convolutional layer of neural network """
    num_examples, height_new, width_new, num_channels_new = dZ.shape
    _, height_prev, width_prev, num_channels_prev = A_prev.shape
    filter_height, filter_width = W.shape[0], W.shape[1]
    num_channels_new_bias = b.shape[3]
    stride_height, stride_width = stride

    if padding == "valid":
        pad_height, pad_width = 0, 0
    elif padding == "same":
        pad_height = int((((height_prev - 1) * stride_height)
                          + filter_height - height_prev) / 2 + 1)
        pad_width = int((((width_prev - 1) * stride_width)
                         + filter_width - width_prev) / 2 + 1)
    else:
        return

    A_prev_padded = np.pad(A_prev, ((0, 0), (pad_height, pad_height),
                                    (pad_width, pad_width),
                                    (0, 0)), mode="constant")

    dA_prev = np.zeros((num_examples, height_prev +
                        (2 * pad_height), width_prev
                        + (2 * pad_width), num_channels_prev))
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(num_examples):
        for channel_idx in range(num_channels_new):
            for j in range(height_new):
                for k in range(width_new):
                    height_start, width_start = ((j * stride_height),
                                                 (k * stride_width))

                    dA_prev[i, height_start: height_start +
                            filter_height, width_start: width_start
                            + filter_width,
                            :] += (
                            dZ[i, j, k, channel_idx] * W[:, :, :, channel_idx]
                    )

                    dW[:, :, :, channel_idx] += (
                            A_prev_padded[i, height_start: height_start
                                          + filter_height,
                                          width_start: width_start
                                          + filter_width, :] *
                            dZ[i, j, k, channel_idx]
                    )

    if padding == "same":
        dA_prev = dA_prev[:, pad_height:-pad_height, pad_width:-pad_width, :]
    return dA_prev, dW, db
