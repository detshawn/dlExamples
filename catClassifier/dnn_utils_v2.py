import numpy as np

RELU = "relu"
SIGMOID = "sigmoid"


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    :param Z: numpy array of any shape

    :return A: output of sigmoid(Z), same shape as Z
    :return cache: return Z as cached data for later usage in backward propagation
    """

    A = 1 / (1 + np.exp(-Z))

    assert(A.shape == Z.shape)

    cache = {"Z": Z}

    return A, cache


def relu(Z):
    """
    Implements the ReLU activation in numpy

    :param Z: numpy array of any shape

    :return A: output of relu(Z), same shape as Z
    :return cache: return Z as cached data for later usage in backward propagation
    """

    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)

    cache = {"Z": Z}

    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implements the backward propagation for a single sigmoid activation in numpy

    :param dA: post-activation gradient, of any shape
    :param cache: Z which we store for computing backward propagation

    :return dZ: Gradient of the cost with respect to Z
    """

    Z = cache
    A, _ = sigmoid(Z)
    dZ = dA * A * (1-A)

    assert (dZ.shape == Z.shape)

    return dZ


def relu_backward(dA, cache):
    """
    Implements the backward propagation for a single ReLU activation in numpy

    :param dA: post-activation gradient, of any shape
    :param cache: Z which we store for computing backward propagation

    :return dZ: Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = dA * (Z >= 0)

    assert (dZ.shape == Z.shape)

    return dZ
