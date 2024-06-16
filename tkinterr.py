import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import save_img

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define function to create image directory and save images
def create_image_directory(images, labels, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for image, label in zip(images, labels):
        filename = os.path.join(directory, f"{label}_{np.random.randint(10000)}.png")
        save_img(filename, image)

# Create directory for test images
test_dir = "test_imag"
create_image_directory(x_test, y_test, test_dir)

