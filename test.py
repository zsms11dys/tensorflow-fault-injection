import numpy as np
import tensorflow as tf
import keras.backend as K
import TensorFI as ti

from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils

CLIP_MIN = -0.5
CLIP_MAX = 0.5
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = load_model("./model/model_mnist(conv).h5")

sess = K.get_session()
fi = ti.TensorFI(sess, logLevel = 100)

results = model.evaluate(x_train, y_train)
print(results)
fi.turnOnInjections()
with sess.as_default():
    results = model.evaluate(x_train, y_train)
print(results)
fi.turnOffInjections()
