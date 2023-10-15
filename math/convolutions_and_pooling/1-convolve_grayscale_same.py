#!/usr/bin/env python3
""" Convolutions & Pooling """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs a same convolution on grayscale images """
    kh, kw = kernel.shape

    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    m, h, w = images.shape
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)])

    output_array = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_section = images[:, i:i + kh, j:j + kw]
            output_array[:, i, j] = np.sum(image_section * kernel, axis=(1, 2))

    return output_array
