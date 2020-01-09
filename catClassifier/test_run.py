from testCases_v4a import *

from parameter_utils import *

# init
print("")
print("--- initialize_parameters_deep() test::")
parameters = initialize_parameters_deep([5,4,3], seed=3)
print(" W1 = " + str(parameters["W1"]))
print(" b1 = " + str(parameters["b1"]))
print(" W2 = " + str(parameters["W2"]))
print(" b2 = " + str(parameters["b2"]))
