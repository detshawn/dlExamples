import numpy as np


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
