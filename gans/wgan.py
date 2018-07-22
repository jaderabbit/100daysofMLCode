from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.initializers import RandomNormal

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class WGAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.depth = 128

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = Adam(lr=0.0002)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        d = self.depth
        my_init = RandomNormal(mean=0.0, stddev=0.02, seed=None)
        model.add(Dense(4 * 4 * d * 8, 
                        activation="relu", 
                        input_shape=(self.latent_dim,),
                        kernel_initializer=my_init))
        model.add(Reshape((4, 4, d*8)))

        model.add(Conv2DTranspose(d*4, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=my_init))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(d*2, kernel_size=5, strides=(2,2), padding='same', kernel_initializer=my_init))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))
       
        model.add(Conv2DTranspose(d, kernel_size=5, strides=(2,2), padding='same', kernel_initializer=my_init))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(self.channels, kernel_size=5, strides=(2,2), padding="same", kernel_initializer=my_init))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        d = self.depth
        model.add(Conv2D(d, kernel_size=4, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(d*2, kernel_size=4, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(d*4, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(d*8, kernel_size=4, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
 
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, data_generator, batch_size=128, sample_interval=50):

        # Load the dataset
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        print("And so it begins...")

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        self.sample_images(-1)
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Select a random batch of images
                imgs, _ = next(data_generator())
                if imgs.shape[0] != batch_size:
                    imgs, _ = next(data_generator())
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = (gen_imgs + 1)*0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("results/albumart_%d.png" % epoch)
        plt.close()

    def save_model(self):
        discriminator_model_json = self.critic.to_json()
        with open("discriminator.json", "w") as json_file:
            json_file.write(discriminator_model_json)
        self.critic.save_weights("discriminator.h5")
        print("Saved discriminator to disk")

        generator_model_json = self.generator.to_json()
        with open("generator.json", "w") as json_file:
            json_file.write(generator_model_json)
        self.generator.save_weights("generator.h5")
        print("Saved generator to disk")

