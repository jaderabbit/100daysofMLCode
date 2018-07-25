from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal

import matplotlib.pyplot as plt


import sys

import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.depth = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
    
        model = Sequential()

        d = self.depth
        my_init = RandomNormal(mean=0.0, stddev=0.02, seed=None)
        model.add(Dense(4 * 4 * d * 8, 
                        activation="linear", 
                        input_shape=(self.latent_dim,),
                        kernel_initializer=my_init))
        model.add(Reshape((4, 4, d*8))) # a vector goes in and a tensor comes.
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(Activation('relu')) 
        # ReLU used commonly in Deep Neural networks as does not suffer so much from vanishing/exploding gradients
        # Better convergence

        model.add(Conv2DTranspose(d*4, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=my_init))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(d*2, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=my_init))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))
       
        model.add(Conv2DTranspose(d, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=my_init))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(self.channels, kernel_size=5, strides=(2,2), padding="same", kernel_initializer=my_init))
        model.add(Activation("tanh")) 
        # According to the original paper:
        ##  We observed that using a bounded activation allowed the model to learn more quickly to saturate
        ##  and cover the color space of the training distribution."
        # This is why we scale the data between -1 and 1 and not 0 and 1 as (-1,1) is the "active" range of tanh

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        d = self.depth

        my_init = RandomNormal(mean=0.0, stddev=0.02, seed=None)
        
        model.add(Conv2D(d, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same", kernel_initializer=my_init))
        model.add(LeakyReLU(alpha=0.2)) 
        # LeakyReLU allows you to set a non zero gradient when unit is not active.
        # I'm not sure I understand why it's needed
        # One source said: "We use a leaky ReLU to allow gradients to flow backwards through the layer unimpeded"
        # Original DCGAN paper said: 
        # => "Within the discriminator we found the leaky rectified activation (Maas et al., 2013) (Xu et al., 2015) to work well, 
        # => especially for higher resolution modeling"


        model.add(Conv2D(d*2, kernel_size=5, strides=2, padding="same", kernel_initializer=my_init))
        model.add(BatchNormalization(momentum=0.9))
        # Best explanation I can find: https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c 
        # The idea: We normalize the input to speed up training, why don't we normalize hidden layers too?
        # => Batch normalization allows each layer of a network to learn by itself a little bit more independently of other layers.
        # => We can use a higher learning rate since high and low values will get "normalized" out
        # => It reduces overfitting because it has a slight regularization effects
        # => Increases stability of the network
        # => batch normalization normalizes the output of a previous activation layer by subtracting the batch mean 
        #    and dividing by the batch standard deviation.

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(d*4, kernel_size=5, strides=2, padding="same", kernel_initializer=my_init))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(d*8, kernel_size=5, strides=1, padding="same", kernel_initializer=my_init))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
 
        model.add(Flatten()) # Takes 3D tensor and flattens it into a single dimensional vector
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.summary() # Just outputs the architecture

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    #TODO Fix epochs
    def train(self, epochs, data_generator, batch_size=128, save_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            imgs, _ = next(data_generator())

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Train it again (different from paper)
            # Make sure discriminator doesn't reach 0
            more_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(more_noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)


    def save_imgs(self, epoch):
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
