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

    elif activation == TANH:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    elif activation == RESUBLU:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = resublu(Z)

    elif activation == HYBRID_RELU_1RESUBLU:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z[:-1, :])
        AL, activation_cache_l = resublu(Z[-1, :].reshape(1,Z.shape[1]))
        A = np.append(A, AL, axis=0)
        activation_cache = {"Z": np.append(activation_cache["Z"], activation_cache_l["Z"], axis=0)}

    else:
        print("ERROR:: " + activation + " function is currently not supported !!")
        exit(1)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = {"linear": linear_cache,
             "activation": activation_cache}

    return A, cache
