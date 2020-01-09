import numpy as np
from dnn_utils_v2 import *


# parameter initialization for 2-layer NN
def initialize_parameters(n_x, n_h, n_y):
    """
    implements param initialization for a two-layer neural network

    :param n_x: input vector size
    :param n_h: # of hidden layer units
    :param n_y: output vector size

    :return parameters:
    W1 -- weight matrix of the hidden layer (n_h, n_x)
    b1 -- bias vector of the hidden layer (n_h, 1)
    W2 -- weight matrix of the output layer (n_y, n_h)
    b2 -- bias vector of the output layer (n_y, 1)
    """

    gain = 0.01
    W1 = np.random.randn(n_h, n_x) * gain
    b1 = np.zeros([n_h, 1])
    W2 = np.random.randn(n_y, n_h) * gain
    b2 = np.zeros([n_y, 1])

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


# parameter initialization for general DNN
def initialize_parameters_deep(layer_dims, seed=None):
    """
    implements param initialization for a general deep neural network

    :param layer_dims: a python array of unit size for each layer

    :return parameters:
    W$l -- weight matrix of the hidden layer (layer_dims[l], layer_dims[l-1])
    b$l -- bias vector of the hidden layer (layer_dims[l], 1)
    """

    np.random.seed(seed=seed)

    parameters = {}
    gain = 0.01
    L = len(layer_dims) - 1  # layer_dims = 1 (input) + L-1 (hidden) + 1 (output)

    for l in range(1, L+1):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])  / np.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros([layer_dims[l], 1])

        assert (parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert (parameters["b" + str(l)].shape == (layer_dims[l], 1))

    return parameters


# parameter update for general DNN
def update_parameters(parameters, grads, learning_rate):
    """
    implements to update parameters given the gradients,
    corresponding to the params, and a learning rate

    :param parameters:
    W$l -- weight matrix of the hidden layer (layer_dims[l], layer_dims[l-1])
    b$l -- bias vector of the hidden layer (layer_dims[l], 1)
    :param grads:
    dW$l -- weight gradient matrix of the hidden layer (layer_dims[l], layer_dims[l-1])
    db$l -- bias gradient vector of the hidden layer (layer_dims[l], 1)
    :param learning_rate: a learning rate factor, double scalar value

    :return parameters: updated parameters
    """

    L = len(parameters) // 2

    for l in range(1, L+1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters
