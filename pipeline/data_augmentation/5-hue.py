#!/usr/bin/env python3
import tensorflow as tf


def change_hue(image, delta):
    """Changes the hue of an image."""
    # Convert the image to float32 for processing
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply the hue change
    altered_image = tf.image.adjust_hue(image, delta)

    # Ensure the image is in the correct range [0, 1]
    altered_image = tf.clip_by_value(altered_image, 0.0, 1.0)

    # Convert back to uint8
    altered_image = tf.image.convert_image_dtype(altered_image, tf.uint8)

    return altered_image
