# Code to demonstrate our two neural network: MFNN and RBFNN

import src.driver as driver
import src.data.data_set as data
from src.networks.radial_basis_nn import RBFNN
from src.networks.mfnn import MFNN
import math
import numpy as np

def test_MFNN():
    # Test Car Data
    np.set_printoptions(precision=2, sign="+", floatmode="fixed", suppress=True, )
    car_classes = ["unacc", "acc", "good", "vgood"]
    car_data = data.get_car_data("../data/car.data", False)
    train, validation = car_data.partition(.8)

    test_example = train.get_data()[0]
    test_example_attribute = test_example[:-1]

    layer = [len(car_data.attr_cols), 5, 4]
    print("Network: " + str(layer))
    print()
    car_network = MFNN(train, validation, layer, 1, 0.1, 100, car_classes)
    car_network.train()
    print()
    print("Test Example: " + str(test_example))
    print("Class to Index: " + str(car_network.class_dict))
    print()
    list_numpy = car_network.get_activation(test_example_attribute)

    for index, activation in enumerate(list_numpy):
        if index == 0:
            print("Input Activation")
            print("\t" + str(activation))
            print("                ")
            print("\t\t\t\t *dot* ")
            print("                ")
            car_network.print_weight(0)
            print("\t\t\t\t *equals*")
            print()
            # for neuron in activation:
            #     print("\t" + str(neuron))
        elif index != 0 and index != (len(list_numpy) - 1):
            print("Hidden Layer Activation")
            print("\t" + str(activation))
            # for neuron in activation:
            #    print("\t" + str(neuron))
        elif index == len(list_numpy) - 1:
            print("Output Activation")
            print("\t" + str(activation))
            # for neuron in activation:
            #   print("\t" + str(neuron))












def main():
    test_MFNN()

main()



