#!/usr/bin/env python3
""" Autoencoders """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates a variational autoencoder """
    encoder_input = keras.layers.Input(shape=(input_dims,))
    encoder_output = encoder_input
    for units in hidden_layers:
        encoder_output = keras.layers.Dense(
            units, activation='relu')(encoder_output)

    mean = keras.layers.Dense(latent_dims)(encoder_output)
    log_sigma = keras.layers.Dense(latent_dims)(encoder_output)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(shape=(
            keras.backend.shape(z_mean)[0], latent_dims), mean=0., stddev=1.)
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    latent_space = keras.layers.Lambda(sampling)([mean, log_sigma])
    encoder = keras.models.Model(
        encoder_input, [mean, log_sigma, latent_space])

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    decoder_output = decoder_input
    for units in reversed(hidden_layers):
        decoder_output = keras.layers.Dense(
            units, activation='relu')(decoder_output)

    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoder_output)
    decoder = keras.models.Model(decoder_input, decoder_output)

    autoencoder = keras.models.Model(
        encoder_input, decoder(encoder(encoder_input)[2]))

    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_input, decoder(encoder(encoder_input)[2])) * input_dims

    kl_loss = (1 + log_sigma -
               keras.backend.square(mean) - keras.backend.exp(log_sigma))
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer='adam')

    return encoder, decoder, autoencoder
