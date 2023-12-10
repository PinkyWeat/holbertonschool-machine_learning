#!/usr/bin/env python3
""" Autoencoders """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ creates a convolutional autoencoder """
    encoder_input = keras.layers.Input(shape=input_dims)
    encoder_output = encoder_input
    for f in filters:
        encoder_output = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(encoder_output)
        encoder_output = keras.layers.MaxPooling2D(
            (2, 2), padding='same')(encoder_output)

    encoder = keras.models.Model(encoder_input, encoder_output)

    decoder_input = keras.layers.Input(shape=latent_dims)
    decoder_output = decoder_input
    for f in reversed(filters[1:]):
        decoder_output = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(decoder_output)
        decoder_output = keras.layers.UpSampling2D((2, 2))(decoder_output)

    decoder_output = keras.layers.Conv2D(
        filters[0], (3, 3), activation='relu')(decoder_output)
    decoder_output = keras.layers.UpSampling2D((2, 2))(decoder_output)
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same',
        activation='sigmoid')(decoder_output)
    decoder = keras.models.Model(decoder_input, decoder_output)

    auto_outputs = encoder(encoder_input)
    auto_outputs = decoder(auto_outputs)
    autoencoder = keras.models.Model(encoder_input, auto_outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
