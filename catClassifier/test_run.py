from testCases_v4a import *

from parameter_utils import *
from forward_utils import *
from cost_utils import *
from backward_utils import *


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


# cost
print("")
print("--- compute_cost() test::")
Y, AL = compute_cost_test_case()

print(" cost = " + str(compute_cost(AL, Y)))
print(" expected cost = " + str(0.2797765635793422))


# backward
print("")
print("--- linear_backward() test::")
# Set up some test inputs
dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print (" dA_prev = "+ str(dA_prev))
print (" dW = " + str(dW))
print (" db = " + str(db))

print (" expected dA_prev = "+ "[[-1.15171336  0.06718465 -0.3204696   2.09812712]")
print("\t\t\t[ 0.60345879 -3.72508701  5.81700741 -3.84326836]")
print("\t\t\t[-0.4319552  -1.30987417  1.72354705  0.05070578]")
print("\t\t\t[-0.38981415  0.60811244 -1.25938424  1.47191593]")
print("\t\t\t[-2.52214926  2.67882552 -0.67947465  1.48119548]]")

print (" expected dW = " + "[[ 0.07313866 -0.0976715  -0.87585828  0.73763362  0.00785716]")
print("\t\t\t[ 0.85508818  0.37530413 -0.59912655  0.71278189 -0.58931808]")
print("\t\t\t[ 0.97913304 -0.24376494 -0.08839671  0.55151192 -0.10290907]]")

print (" expected db = " + "[[-0.14713786]")
print("\t\t\t[-0.11313155]")
print("\t\t\t[-0.13209101]]")

print("")
print("--- linear_activation_backward() test::")
dAL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
print (" sigmoid:")
print ("  dA_prev = "+ str(dA_prev))
print ("  dW = " + str(dW))
print ("  db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
print (" relu:")
print ("  dA_prev = "+ str(dA_prev))
print ("  dW = " + str(dW))
print ("  db = " + str(db))

print("")
print("--- L_model_backward() test::")
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)
