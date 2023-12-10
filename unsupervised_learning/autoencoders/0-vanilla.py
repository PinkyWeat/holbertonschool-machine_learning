#!/usr/bin/env python3
""" Autoencoders """
import tensorflow.keras as keras


# input_dims is an integer containing the dimensions of the model input
# hidden_layers is a list containing the number of nodes for each
# hidden layer in the encoder, respectively
# the hidden layers should be reversed for the decoder
# latent_dims is an integer containing the dimensions of the latent
# space representation
# Returns: encoder, decoder, auto
# encoder is the encoder model
# decoder is the decoder model
# auto is the full autoencoder model
# The autoencoder model should be compiled using adam
# optimization and binary cross-entropy loss
# All layers should use a relu activation except for the last
# layer in the decoder, which should use sigmoid
def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates an autoencoder """
    encoder_input = keras.layers.Input(shape=(input_dims,))
    encoder_output = encoder_input
    for units in hidden_layers:
        encoder_output = keras.layers.Dense(
            units, activation='relu')(encoder_output)

    latent_space = keras.layers.Dense(
        latent_dims, activation='relu')(encoder_output)
    encoder = keras.models.Model(encoder_input, latent_space)

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    decoder_output = decoder_input
    for units in reversed(hidden_layers):
        decoder_output = keras.layers.Dense(
            units, activation='relu')(decoder_output)

    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoder_output)
    decoder = keras.models.Model(decoder_input, decoder_output)

    auto_outputs = encoder(encoder_input)
    auto_outputs = decoder(auto_outputs)
    autoencoder = keras.models.Model(encoder_input, auto_outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
