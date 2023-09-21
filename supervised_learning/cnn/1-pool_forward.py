#!/usr/bin/env python3
""" Convolutional Neural Network """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs frwd prop over pooling layer of a neural network """
    num_samples, prev_height, prev_width, prev_channels = A_prev.shape
    filter_height, filter_width = kernel_shape
    stride_height, stride_width = stride

    output_height = int((prev_height - filter_height) / stride_height) + 1
    output_width = int((prev_width - filter_width) / stride_width) + 1

    output_img = np.zeros((num_samples, output_height,
                           output_width, prev_channels))

    for i in range(output_height):
        x = i * stride_height

        for j in range(output_width):
            y = j * stride_width
            pool_slice = A_prev[:, x: x + filter_height,
                                y: y + filter_width, :]

            if mode == "max":
                result = np.max(pool_slice, axis=(1, 2))
            elif mode == "avg":
                result = np.mean(pool_slice, axis=(1, 2))
            output_img[:, i, j, :] = result

    return output_img
