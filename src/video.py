# Code to demonstrate our two neural network: MFNN and RBFNN

import src.driver as driver
import src.data.data_set as data
from src.networks.radial_basis_nn import RBFNN
from src.networks.mfnn import MFNN
import math
import numpy as np


def get_prediction(example, class_dict):
    max_index = 0
    for i in range(1, len(example)):
        if example[max_index] < example[i]:
            max_index = i

    for key in class_dict:
        value = class_dict[key]
        if value == max_index:
            return key


def test_RBFNN():
    np.set_printoptions(precision=2, sign="+", floatmode="fixed", suppress=True, )
    fires_center_data = data.get_machine_data("../rbf-data/machine-kmeans-fold-0-centers.txt", False)
    fires_training_data = data.get_machine_data("../rbf-data/machine-kmeans-fold-0-train.txt", False)
    fires_training_data, fires_validation_data = fires_training_data.partition(.8)
    fires_test_data = data.get_machine_data("../rbf-data/machine-kmeans-fold-0-test.txt", False)
    num_inputs = len(fires_center_data.attr_cols)

    test_example = fires_test_data.data[0]

    test_example_rounded = []
    for col in test_example:
        if col is float:
            test_example_rounded.append(round(col*100)/100)
        test_example_rounded.append(col)
    test_example_rounded = test_example_rounded

    fire_network = RBFNN(fires_center_data, fires_training_data, fires_validation_data, num_inputs, .5, 100)

    print()
    print("Center vectors for hidden nodes: ")
    print()
    fire_network.print_centers()
    print()
    print()

    fire_network.train()

    input_activation = []
    for col in fires_center_data.attr_cols:
        input_activation.append(test_example[col])
    input_activation.append(1/(1+math.exp(-1)))
    input_activation = np.array(input_activation)
    output_activation, hidden_activation = fire_network.run_rbfnn(test_example)
    print()
    print("Test Example: " + str(test_example_rounded))
    print("Expected output: " + str(test_example_rounded[-1]))
    print()
    print("The input activation:")
    print(input_activation)
    print()
    print("")
    print("Compared to the following center vectors of each hidden node...")
    print()
    fire_network.print_centers()
    print()
    print("Equals the activation of the hidden layer:")
    print()
    print(hidden_activation)
    print()
    print("Combined via the dot product with the weight matrix:")
    print()
    fire_network.print_weight(0)
    print()
    print("And, finally, sigmoided using a logistic function...")
    print()
    print("Equals the output activation:")
    print(output_activation)
    print()
    print("Which tells us that the predicted value is: ")
    print()
    print(output_activation[0])
    print()


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
    print()
    list_numpy = car_network.get_activation(test_example_attribute)

    output_activation = None
    for index, activation in enumerate(list_numpy):
        if index == 0:
            print()
            print("The input activation:")
            print(activation)
            print()
            print("Combined via the dot product with the first weight matrix:")
            print()
            car_network.print_weight(0)
            print()
            print("And, finally, sigmoided using a logistic function...")
            print()
            # for neuron in activation:
            #     print("\t" + str(neuron))
        elif index != 0 and index != (len(list_numpy) - 1):
            print("Equals the hidden layer's activation:")
            print(activation)
            print()
            print("Combined via the dot product with the second weight matrix:")
            print()
            car_network.print_weight(1)
            print()
            print("And, finally, sigmoided using a logistic function...")
            print()
        elif index == len(list_numpy) - 1:
            print("Equals the output activation:")
            print(activation)
            output_activation = activation
    print()
    print("Which we then produce a predicted class using the following index mappings: ")
    print()
    print(car_network.class_dict)
    print()
    print("Which tells us that the predicted class is: ")
    print()
    print(get_prediction(output_activation, car_network.class_dict))
    print()


def main():
    # test_MFNN()
    test_RBFNN()


main()



