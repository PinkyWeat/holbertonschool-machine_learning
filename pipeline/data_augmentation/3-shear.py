#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import apply_affine_transform

def shear_image(image, intensity):
    """Randomly shears an image while preserving color."""
    # Convert the image to a numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)

    # Apply the shear transformation without specifying axes
    sheared_image = apply_affine_transform(image, shear=intensity)

    # Ensure the image is in the correct range [0, 255]
    sheared_image = np.clip(sheared_image, 0, 255)

    # Convert back to a tensor and ensure the data type is correct
    sheared_image = tf.convert_to_tensor(sheared_image, dtype=tf.uint8)

    return sheared_image
