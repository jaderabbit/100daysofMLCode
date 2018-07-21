from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class WGAN():
    def __init__(self, rows, cols, channels):
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss, 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=self.wasserstein_loss, optimizer='adam')

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss, 
            optimizer=optimizer,
            metrics=['accuracy'])

        

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        noise_shape = (100,)
        
        model = Sequential()

        d = 128
        
        model.add(Dense(4 * 4 * d * 8, activation="linear", input_shape=noise_shape))
        model.add(Reshape((4, 4, d*8)))

        model.add(Conv2DTranspose(d*4, 4, strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        
        model.add(Conv2DTranspose(d*2, 4, strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
       
        model.add(Conv2DTranspose(d, 4, strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(d, 4, strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(3, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()
        d = 128
        model.add(Conv2D(d, kernel_size=4, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(d*2, kernel_size=4, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(d*4, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(d*8, kernel_size=4, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.summary()

        img = Input(shape=img_shape)
        features = model(img)
        valid = Dense(1, activation="linear")(features)

        return Model(img, valid)

    def train(self, epochs, data_generator, batch_size=128, save_interval=50):

        # Rescale -1 to 1
        

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                imgs, things = next(data_generator())
                imgs = imgs[0:half_batch]
                noise = np.random.normal(0, 1, (half_batch, 100))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict(noise)
                print(gen_imgs.shape)
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, -np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip discriminator weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, -np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("results/albumart_%d.png" % epoch)
        plt.close()

    def save_model(self):
        discriminator_model_json = self.discriminator.to_json()
        with open("discriminator.json", "w") as json_file:
            json_file.write(discriminator_model_json)
        self.discriminator.save_weights("discriminator.h5")
        print("Saved discriminator to disk")

        generator_model_json = self.generator.to_json()
        with open("generator.json", "w") as json_file:
            json_file.write(generator_model_json)
        self.generator.save_weights("generator.h5")
        print("Saved generator to disk")

