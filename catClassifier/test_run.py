from testCases_v4a import *

from parameter_utils import *
from forward_utils import *

# init
print("")
print("--- initialize_parameters_deep() test::")
parameters = initialize_parameters_deep([5,4,3], seed=3)
print(" W1 = " + str(parameters["W1"]))
print(" b1 = " + str(parameters["b1"]))
print(" W2 = " + str(parameters["W2"]))
print(" b2 = " + str(parameters["b2"]))


# forward
print("")
print("--- linear_forward() test::")
A, W, b = linear_forward_test_case()

Z, linear_cache = linear_forward(A, W, b)
print(" Z = " + str(Z))

print(" expected Z = " + "[[ 3.26295337 -1.23429987]]")

print("")
print("--- linear_activation_forward() test::")
A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print(" With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print(" With ReLU: A = " + str(A))

print(" expected A w/ sigmoid = " + "[[ 0.96890023 0.11013289]]")
print(" expected A w/ ReLU = " + "[[ 3.43896131 0. ]]")

print("")
print("--- L_model_forward() test::")
X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print(" AL = " + str(AL))
print(" Length of caches list = " + str(len(caches)))
print(" expected AL = " + "[[0.03921668 0.70498921 0.19734387 0.04728177]]")
print(" expected Len of caches list = " + str(3))
