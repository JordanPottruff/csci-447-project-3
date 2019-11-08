# Code to demonstrate our two neural network: MFNN and RBFNN

import src.driver as driver
import src.data.data_set as data
from src.networks.radial_basis_nn import RBFNN
from src.networks.mfnn import MFNN
import math

def test_MFNN():
    # Test Car Data
    car_classes = ["unacc", "acc", "good", "vgood"]
    car_data = data.get_car_data("../data/car.data", False)
    train, validation = car_data.partition(.8)

    test_example = train.get_data()[0]
    test_example_attribute = test_example[:-1]

    layer = [len(car_data.attr_cols), 5, 4]
    car_network = MFNN(train, validation, layer, 1, 0.1, 100, car_classes)
    car_network.train()
    print()
    print("Network: " + str(layer))
    print("Test Example: " + str(test_example))
    print(car_network.class_dict)
    list_numpy = car_network.get_activation(test_example_attribute)

    for index, activation in enumerate(list_numpy):
        if index == 0:
            print("Input Activation")
            for neuron in activation:
                print("\t" + str(neuron))
        elif index != 0 and index != (len(list_numpy) - 1):
            print("Hidden Layer Activation")
            for neuron in activation:
                print("\t" + str(neuron))
        elif index == len(list_numpy) - 1:
            print("Output Activation")
            for neuron in activation:
                print("\t" + str(neuron))












def main():
    test_MFNN()

main()



