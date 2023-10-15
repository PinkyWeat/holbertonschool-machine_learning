#!/usr/bin/env python3
""" Convolutions & Pooling """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ performs a valid convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_height = h - kh + 1
    out_weight = w - kw + 1

    out_array = np.zeros((m, out_height, out_weight))

    for i in range(out_height):
        for j in range(out_weight):
            image_section = images[:, i:i+kh, j:j+kw]
            out_array[:, i, j] = np.sum(image_section * kernel, axis=(1,2))

    return out_array
