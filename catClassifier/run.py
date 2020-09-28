import numpy as np
import matplotlib.pyplot as plt
# import imageio
from PIL import Image
# from imageio import imread
from lr_utils import load_data

from L_model import *
from L_bi_model import *
from dnn_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# data import
train_x_orig, train_y, test_x_orig, test_y, classes = load_data('catvnoncat')

# example of a picture
index = 10
# plt.imshow(train_x_orig[index])
print("=> example of a picture")
print (" y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
# input ("Press Enter to continue...")

# explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("=> explore your dataset!")
print (" Number of training examples: " + str(m_train))
print (" Number of testing examples: " + str(m_test))
print (" Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print (" train_x_orig shape: " + str(train_x_orig.shape))
print (" train_y shape: " + str(train_y.shape))
print (" test_x_orig shape: " + str(test_x_orig.shape))
print (" test_y shape: " + str(test_y.shape))

# pre-processing
train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255.
test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255.

print (" train_x's shape: " + str(train_x.shape))
print (" test_x's shape: " + str(test_x.shape))


# model definition
layers_dims = [train_x.shape[0], 20, 7, 5, 1] #  4-layer model
print(" layers_dims: "+str(layers_dims))

# model learning
parameters = L_bi_layer_model(train_x, train_y, layers_dims,
              num_iterations = 2500, print_cost = True)

# prediction
train_accuracy, train_p, _ = predict_bi(train_x, train_y, parameters)
test_accuracy, test_p, _ = predict_bi(test_x, test_y, parameters)
print(parameters)

# another test images
my_image = "cat.7.jpg"
fname = "datasets/" + my_image
image = Image.open(fname)
# image.show()
image = image.resize((num_px, num_px), Image.ANTIALIAS)
my_image = np.array(image).reshape(1, num_px*num_px*3).T
my_image = my_image / 255.
# print(my_image.shape)
my_predicted_image, prob, _ = predict_bi(my_image, np.ones([1,1]), parameters)

# plotting
print("=> prediction result")
print(" training accuracy = {} %".format(train_accuracy*1e2))
print(" test accuracy = {} %".format(test_accuracy*1e2))

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
      classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
