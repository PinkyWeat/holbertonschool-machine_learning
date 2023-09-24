#!/usr/bin/env python3
""" Convolutional Neural Network """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs back prop over pooling layer of neural network """
    num_examples, height_new, width_new, num_channels_new = dA.shape
    _, height_prev, width_prev, num_channels_prev = A_prev.shape
    kernel_height, kernel_width = kernel_shape
    stride_height, stride_width = stride

    dA_prev = np.zeros_like(A_prev)

    # Backpropagation!
    for example in range(num_examples):
        for height in range(height_new):
            for width in range(width_new):
                for channel in range(num_channels_prev):
                    grad_value = dA[example, height, width, channel]

                    # them' corners of current slice
                    vertical_start = height * stride_height
                    vertical_end = vertical_start + kernel_height
                    horizontal_start = width * stride_width
                    horizontal_end = horizontal_start + kernel_width

                    # gradients depending on the pooling mode
                    if mode == "max":
                        # finds position of its maximum value in the window
                        window_slice = A_prev[example, vertical_start:
                                              vertical_end,
                                              horizontal_start:horizontal_end,
                                              channel]
                        mask = (window_slice == np.max(window_slice))
                    elif mode == "avg":
                        # compute the mask for average pooling
                        mask = (np.ones((kernel_height, kernel_width))
                                / (kernel_height * kernel_width))
                    else:
                        return

                    dA_prev[example, vertical_start:vertical_end,
                            horizontal_start:horizontal_end, channel] += (
                                mask * grad_value)

    return dA_prev
