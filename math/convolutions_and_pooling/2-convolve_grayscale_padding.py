#!/usr/bin/env python3
""" Convolutions & Pooling """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ performs a convolution on grayscale images with custom padding """
    ph, pw = padding
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)])
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_height = h - kh + 1
    out_weight = w - kw + 1

    output_array = np.zeros((m, out_height, out_weight))

    for i in range(out_height):
        for j in range(out_weight):
            image_section = images[:, i:i + kh, j:j + kw]
            output_array[:, i, j] = np.sum(image_section * kernel, axis=(1, 2))

    return output_array