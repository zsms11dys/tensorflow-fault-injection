import argparse

import numpy as np
import tensorflow as tf
import keras.backend as K
import TensorFI as ti

from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l2

CLIP_MIN = -0.5
CLIP_MAX = 0.5

def train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    layers = [
        Dense(256, input_shape = (784,)),
        Activation("relu"),
        Dense(256),
        Activation("relu"),
        Dense(128),
        Activation("relu"),
        Dense(128),
        Activation("relu"),
        Dense(10),
        Activation("softmax"),
    ]

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model = Sequential()
    for layer in layers:
        model.add(layer)
    
    model.compile(
        loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
    )

    K.get_session().run(tf.global_variables_initializer())
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test),
    )
    
    print(model.summary())

    model.save("./model/model_mnist.h5")
    
    loss, acc = model.evaluate(x_test, y_test)
    
    print(acc)
    #0.976


if __name__ == "__main__":
    train()
