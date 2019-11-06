import numpy as np
import random
import math
import src.data.data_set as data


class MFNN:
    def __init__(self, training_data, validation_data, layer_size, learning_rate, momentum, classes=None):
        self.training_data = training_data
        self.validation_data = validation_data
        self.num_layers = len(layer_size)
        self.layer_size = layer_size
        self.weights = self.init_weights()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.class_dict = None if classes is None else {cls: index for index, cls in enumerate(classes)}

    def is_regression(self):
        return self.class_dict is None

    # The output of the neural network is a numpy array, rather than a single value. In fact, the output is just the
    # activations of the final layer, where either each node corresponds to the probability of a class (classification),
    # or there is only one output node which signifies a regression estimate (regression). This function is therefore
    # required for determining the "expected" output array for a given expected class.
    def get_class_array(self, class_value):
        if self.is_regression():
            return np.array([class_value])
        else:
            class_index = self.class_dict[class_value]
            num_classes = len(self.class_dict)
            # Return an array with all zeros except for the position representing the class_value.
            class_array = np.zeros(num_classes)
            class_array[class_index] = 1
            return class_array

    # Returns the value of the class with the highest activation in the given class array. See 'get_class_array' for a
    # better understanding of the purpose of these functions.
    def get_class_value(self, class_array):
        if self.is_regression():
            return class_array[0]
        else:
            inverted_class_dict = {value: key for key, value in self.class_dict.items()}
            max_index = 0
            for i in range(1, len(class_array)):
                if class_array[i] > class_array[max_index]:
                    max_index = i
            return inverted_class_dict[max_index]

    def train(self):
        numpy_training_data = self.training_data.get_numpy_list()

        mini_batch_size = 4
        for i in range(100000):
            # print(self.weights)
            if self.is_regression():
                print("Error: " + str(self.get_validation_error()))
            else:
                print("Accuracy: " + str(self.get_validation_accuracy()))
            random.shuffle(numpy_training_data)
            mini_batches = [numpy_training_data[k:k + mini_batch_size] for k in range(0, len(numpy_training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch)

    def train_mini_batch(self, mini_batch):
        for example_array, expected_class in mini_batch:
            expected_array = self.get_class_array(expected_class)
            delta_weights = self.backpropagation(example_array, expected_array)
            for i in range(len(self.weights)):
                self.weights[i] -= (self.learning_rate/len(mini_batch)) * delta_weights[i]

    def get_validation_error(self):
        numpy_validation_data = self.validation_data.get_numpy_list()
        squared_sum = 0
        for example_array, expected_class in numpy_validation_data:
            output = self.run(example_array)
            squared_sum += (self.get_class_value(output) - expected_class)**2
        return math.sqrt(squared_sum)

    def get_validation_accuracy(self):
        numpy_validation_data = self.validation_data.get_numpy_list()
        correct = 0
        for example_array, expected_class in numpy_validation_data:
            output = self.run(example_array)
            if self.get_class_value(output) == expected_class:
                correct += 1
        return correct / len(numpy_validation_data)

    def init_weights(self):
        weights = []
        for size in range(1, len(self.layer_size)):
            weights.append(np.random.randn(self.layer_size[size], self.layer_size[size-1] + 1))
        return weights

    def get_activation(self, example: np.ndarray):
        inputs = np.append(example, sigmoid(1))
        activation = [inputs]
        for i in range(len(self.weights)):
            inputs = np.dot(self.weights[i], inputs)
            if not (self.is_regression() and i == len(self.weights) - 1):
               inputs = sigmoid(inputs)
            if i < len(self.weights) - 1:
                inputs = np.append(inputs, sigmoid(1))
            activation.append(inputs)
        return activation

    def backpropagation(self, example: np.ndarray, expected: np.ndarray):
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        n = len(self.layer_size)

        # Feedforward
        activation = self.get_activation(example)

        # Output Layer
        delta = cost_prime(expected, activation[-1])
        if not self.is_regression():
            delta = delta * sigmoid_prime(activation[-1])
        delta_weights[-1] = np.outer(delta, activation[-2])

        # Hidden Layer
        for i in range(n-2, 0, -1):
            previous_activiation = activation[i-1]
            downstream_weights = self.weights[i].T

            if i < n-2:
                delta = delta[:-1]
            delta = np.dot(downstream_weights, delta) * sigmoid_prime(activation[i])
            delta_weights[i-1] = np.delete(np.outer(delta, previous_activiation), -1, 0)
        return delta_weights

    def run(self, example):
        return self.get_activation(example)[-1]


def sigmoid_prime(sigmoid):
    return sigmoid*(1-sigmoid)


def cost_prime(expected, actual):
    return actual - expected


def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-x))





















