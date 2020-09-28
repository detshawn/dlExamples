import numpy as np

RELU = "relu"
SIGMOID = "sigmoid"
TANH = "tanh"
RESUBLU = "resublu"
HYBRID_RELU_1RESUBLU = "hybrid_relu_1resublu"


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


def tanh(Z):
    """
    Implements the Tanh Unit activation in numpy

    :param Z: numpy array of any shape

    :return A: output of tanh(Z), same shape as Z
    :return cache: return Z as cached data for later usage in backward propagation
    """

    SIG_Z, _ = sigmoid(2*Z)
    A = 2 * SIG_Z - 1

    assert(A.shape == Z.shape)

    cache = {"Z": Z}

    return A, cache


def resublu(Z):
    """
    Implements the Rectified Sublinear Unit activation in numpy

    :param Z: numpy array of any shape

    :return A: output of resublu(Z), same shape as Z
    :return cache: return Z as cached data for later usage in backward propagation
    """
    TANH_Z, _ = tanh(Z)
    A = np.maximum(0, TANH_Z)

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


def tanh_backward(dA, cache):
    """
    Implements the backward propagation for a single TanH activation in numpy

    :param dA: post-activation gradient, of any shape
    :param cache: Z which we store for computing backward propagation

    :return dZ: Gradient of the cost with respect to Z
    """

    Z = cache
    A, _ = tanh(Z)
    dZ = dA * (1+A) * (1-A)

    assert (dZ.shape == Z.shape)

    return dZ


def resublu_backward(dA, cache):
    """
    Implements the backward propagation for a single ReSUBLU activation in numpy

    :param dA: post-activation gradient, of any shape
    :param cache: Z which we store for computing backward propagation

    :return dZ: Gradient of the cost with respect to Z
    """

    Z = cache
    A, _ = tanh(Z)
    detA = (1+A) * (1-A)
    ReA = detA * (A >= 0)
    dZ = dA * ReA

    assert (dZ.shape == Z.shape)

    return dZ


def compute_cost(AL, Y):
    """
    implements the cost computation for a neural network

    :param AL: probability vector, numpy array (1, num_samples)
    :param Y: true value vector of NN model in supervised learning, numpy array (1, num_samples

    :return: computed cost, float scalar value
    """

    m = Y.shape[1]

    cost = (-1/m) * (np.dot(np.log(AL), Y.T) + np.dot(np.log(1-AL), (1-Y).T))
    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost
