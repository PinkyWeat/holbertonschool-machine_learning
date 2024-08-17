#!/usr/bin/env python3
import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly changes the brightness of an image."""
    # Convert the image to float32 for processing
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply random brightness change
    brightened_image = tf.image.random_brightness(image, max_delta=max_delta)

    # Ensure the image is in the correct range [0, 1]
    brightened_image = tf.clip_by_value(brightened_image, 0.0, 1.0)

    # Convert back to uint8
    brightened_image = tf.image.convert_image_dtype(brightened_image, tf.uint8)

    return brightened_image
