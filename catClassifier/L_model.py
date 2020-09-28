import numpy as np
import matplotlib.pyplot as plt
from parameter_utils import *
from forward_utils import *
from cost_utils import *
from backward_utils import *


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


def L_layer_model(X, Y, layers_dims,
                  learning_rate=0.0075, num_iterations=3000, print_cost=False, print_plot=False):

    np.random.seed(1)
    costs = []

    # parameter init
    parameters = initialize_parameters_deep(layers_dims, seed=1)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    if print_plot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


def predict(X, Y, parameters):
    """
    implements to predict the classification probability using the input X and trained parameters

    :param X: input data, numpy matrix
    :param parameters: trained parameters, consisting of Ws and bs

    :return accuracy: accuracy rate of prediction to the true label vector Y, numpy array of 0/1's
    :return y_prediction: predicted value based on the probability vector AL, range in [0,1]
    :return caches: cached data for forward propagation in prediction
    """

    L = len(parameters) // 2
    AL, caches = L_model_forward(X, parameters)

    Y_prediction = (AL > 0.5)
    accuracy = 1-np.mean(np.abs(Y_prediction - Y))

    return accuracy, Y_prediction, caches
