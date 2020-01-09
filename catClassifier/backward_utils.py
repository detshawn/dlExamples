import numpy as np
from dnn_utils_v2 import *


def linear_backward(dZ, cache):
    """
    implements the gradient calculation for dW and db given dZ and A_prev

    :param dZ: the gradient of the cost with respect to Z
    :param cache: linear cache (A_rev, W, b)

    :return dA_prev: the gradient of the cost with respect to the activation
    :return dW: the gradient of the cost with respect to W
    :return db: the gradient of the cost with respect to b
    """

    A_prev = cache["A_prev"]
    W = cache["W"]
    b = cache["b"]
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    implements per-layer backward propagation

    :param dA: the gradient of the cost with respect to the activation
    :param cache: cached data from linear & activation processes in forward propagation
    :param activation: activation function type, string

    :return dA_prev: the gradient of the cost w.r.t. the previous activation
    :return dW: the gradient of the cost with respect to W
    :return db: the gradient of the cost with respect to b
    """

    linear_cache = cache["linear"]
    activation_cache = cache["activation"]

    if activation == RELU:
        dZ = relu_backward(dA, activation_cache["Z"])
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == SIGMOID:
        dZ = sigmoid_backward(dA, activation_cache["Z"])
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    else:
        exit(1)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    implements the backward propagation for L-layer DNN model

    :param AL: probability vector, output of the forward propagation
    :param Y: true value vector in supervised learning
    :param caches: list of caches for linear & activation processes in forward propagation

    :return grads:
    """

    grads = {}
    L = len(caches)

    # output layer with sigmoid function
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, caches[L-1], SIGMOID)

    # hidden layers in reversed iteration
    for l in reversed(range(0, L-1)):
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l+1)], caches[l], RELU)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads
