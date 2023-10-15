#!/usr/bin/env python3
""" Convoluutions & Pooling """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ performs a convolution on images using multiple kernels """
    kh, kw, _, nc = kernels.shape
    m, h, w, _ = images.shape
    sh, sw = stride
    ph, pw = 0, 0

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1

    if pw and ph:
        images = np.pad(images, [(0, 0), (ph, ph), (pw, pw), (0, 0)])

    out_height = ((h + (2 * ph) - kh) // sh) + 1
    out_weight = ((w + (2 * pw) - kw) // sw) + 1

    output_array = np.zeros((m, out_height, out_weight, nc))

    for k in range(nc):
        kernel = kernels[:, :, :, k]
        for i in range(out_height):
            for j in range(out_weight):
                image_section = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
                output_array[:, i, j, k] = np.sum(image_section *
                                                  kernel, axis=(1, 2, 3))

    return output_array
