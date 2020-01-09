import numpy as np
import matplotlib.pyplot as plt
from parameter_utils import *
from forward_utils import *
from cost_utils import *
from backward_utils import *


def L_layer_model(X, Y, layers_dims,
                  learning_rate=0.0075, num_iterations=3000, print_cost=False):

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
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    input("Press Enter to continue...")


    return parameters
