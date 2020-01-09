import numpy as np
from dnn_utils_v2 import *


def linear_forward(A_prev, W, b):
    """
    implements a linear calculation for a single layer characterized by W and b, given A_prev

    :param A_prev: the output from the previous layer
                   (or the input matrix X if it is the first hidden layer),
                   any size of numpy array
    :param W: a weight matrix for the given layer, numpy matrix
    :param b: a bias vector for the given layer, numpy array

    :return Z: the output of linear calculation
    :return cache: A_prev, W and b for later usage in backward propagation
    """

    Z = np.dot(W, A_prev) + b

    assert (Z.shape == (W.shape[0], A_prev.shape[1]))

    cache = {"A_prev": A_prev,
             "W": W,
             "b": b}

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    implements the neuron and following activation calculation
    for a single layer characterized by W, b and activation function, given A_prev

    :param A_prev: the output from the previous layer
                   (or the input matrix X if it is the first hidden layer),
                   any size of numpy array
    :param W: a weight matrix for the given layer, numpy matrix
    :param b: a bias vector for the given layer, numpy array
    :param activation: activation function name in string

    :return A: the output of activation for a single layer
    :return cache: linear and activation cache (W, b / Z) for later usage in backward propagation
    """

    if activation == RELU:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == SIGMOID:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    else:
        print("ERROR:: " + activation + " function is currently not supported !!")
        exit(1)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = {"linear": linear_cache,
             "activation": activation_cache}

    return A, cache


def L_model_forward(X, parameters):
    """
    implements the forward propagation of a neural network

    :param X: input data samples, any size of numpy matrix (n_x, num_samples)
    :param parameters:
    Wl -- weight matrix of l-th layer (layer_dims[l], layer_dims[l-1)
    bl -- bias matrix of l-th layer (layer_dims[l], 1)

    :return AL: result of the (L-1)-th layer, or output layer, numpy array (n_y, num_samples)
    :return caches: list of caches for each layer calculation
    """

    # init
    caches = []
    A = X
    L = len(parameters) // 2

    # iterations
    for l in range(1, L): # L-1 hidden layers with ReLU activation
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                  parameters["W" + str(l)],
                                  parameters["b" + str(l)],
                                  RELU)

        caches.append(cache)

    # output layer with a sigmoid activation
    AL, cache = linear_activation_forward(A,
                                  parameters["W" + str(L)],
                                  parameters["b" + str(L)],
                                  SIGMOID)

    assert (AL.shape == (parameters["W" + str(L)].shape[0], X.shape[1]))
    caches.append(cache)

    return AL, caches
