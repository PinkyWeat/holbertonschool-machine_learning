#!/usr/bin/env python3
""" Convoluutions & Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling on images """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_height = ((h - kh) // sh) + 1
    out_weight = ((w - kw) // sw) + 1

    output_array = np.zeros((m, out_height, out_weight, c))

    for i in range(out_height):
        for j in range(out_weight):
            image_section = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            if mode == 'max':
                output_array[:, i, j, :] = np.max(image_section, axis=(1, 2))
            if mode == 'avg':
                output_array[:, i, j, :] = np.average(image_section,
                                                      axis=(1, 2))

    return output_array
