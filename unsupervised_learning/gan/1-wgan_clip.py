import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):

    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        # Define the generator loss and optimizer:
        self.generator.loss = lambda fake_output: -tf.math.reduce_mean(fake_output)
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)

        # Define the discriminator loss and optimizer:
        self.discriminator.loss = lambda real_output, fake_output: tf.math.reduce_mean(fake_output) - tf.math.reduce_mean(real_output)
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)


    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, _):
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_images = self.get_real_sample()
                fake_images = self.get_fake_sample(training=True)

                real_output = self.discriminator(real_images, training=True)
                fake_output = self.discriminator(fake_images, training=True)

                disc_loss = self.discriminator.loss(real_output, fake_output)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            # Clip weights
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1, 1))

        with tf.GradientTape() as gen_tape:
            generated_images = self.get_fake_sample(training=True)
            gen_output = self.discriminator(generated_images, training=True)
            gen_loss = self.generator.loss(gen_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return {"discr_loss": disc_loss, "gen_loss": gen_loss}

