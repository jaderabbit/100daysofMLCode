import keras
from keras.models import load_model
from keras.models import Sequential
from PIL import Image
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from gans import wgan

batch_size = 32

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        r'dataset/audio-covers',  # this is the target directory
        target_size=(64, 64),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary',
        save_to_dir=r'dataset/audio-covers-processed',
        save_format='jpeg')  # since we use binary_crossentropy loss, we need binary labels

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

w = wgan.WGAN(64, 64, 3)
w.train(epochs=10000, train_generator=train_generator, batch_size=32, save_interval=100)
w.save_model()
