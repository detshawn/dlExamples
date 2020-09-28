from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from io_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from HappyModel import *


# data import
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("# training exps = {}".format(X_train.shape[0]))
print("# test exps = {}".format(X_test.shape[0]))
print(" X_train shape = {}".format(X_train.shape))
print(" Y_train shape = {}".format(Y_train.shape))
print(" X_test shape = {}".format(X_test.shape))
print(" Y_test shape = {}".format(Y_test.shape))

# create the model
happyModel = HappyModel(X_train.shape[1:])
# plot_model(happyModel, to_file='HappyModel.png')

# compile the model
happyModel.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=["accuracy"])

# train the model
hist = happyModel.fit(x=X_train, y=Y_train,
                      validation_data=(X_test, Y_test),
                      batch_size=16, epochs=10)
# print(hist.history['loss'])
# print(hist.history['accuracy'])

# evaluate the model
preds = happyModel.evaluate(x=X_test, y=Y_test)
print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

# result
happyModel.summary()
# SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
print_and_save_hist(hist)
