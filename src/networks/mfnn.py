import numpy as np
import random
import math
import src.activation_functions as af

# The amount each new average metric needs to be better than the old average metric for the training process to
# continue.
CONVERGENCE_THRESHOLD = .0001


class MFNN:
    def __init__(self, training_data, validation_data, layer_size, learning_rate, momentum, convergence_size, classes=None):
        self.training_data = training_data
        self.validation_data = validation_data
        self.num_layers = len(layer_size)
        self.layer_size = layer_size
        self.weights = self.init_weights()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.convergence_size = convergence_size
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
        convergence_check = []

        # Each cycle of this while loop is a single epoch. That is, it covers all of the training data (shuffled in
        # random order and put into mini batches). This loop will break based on the convergence calculation used
        # immediately below.
        count = 0
        while True:
            count += 1
            # Check for convergence by evaluating the past self.convergence_size*2 validation metrics (either accuracy
            # or error). We exit if the older half of metrics has a better average than the newer half.
            metric = self.get_error(self.validation_data) if self.is_regression() else \
                self.get_accuracy(self.validation_data)
            convergence_check.append(metric)
            # Wait until the convergence check list has all self.convergence_size*2 items.
            if len(convergence_check) > self.convergence_size*2:
                # Remove the oldest metric, to maintain the list's size.
                convergence_check.pop(0)
                # The last half of the list are the older metrics.
                old_metric = sum(convergence_check[:self.convergence_size])
                # The first half of the list are the newer metrics.
                new_metric = sum(convergence_check[self.convergence_size:])
                # We compare the difference in sums. We could use averages, but there is no difference when comparing
                # the sums or averages since the denominator would be the same size for both.
                difference = new_metric - old_metric
                if count % 200:
                    print("Accuracy so far..." + "{:.2f}".format(new_metric))
                if self.is_regression():
                    # Error needs to invert the difference, as we are MINIMIZING error.
                    if -difference < CONVERGENCE_THRESHOLD:

                        return
                else:
                    # We attempt to MAXIMIZE accuracy for classification data.
                    if difference < CONVERGENCE_THRESHOLD:
                        print("Final Validation Accuracy: " + "{:.2f}".format(new_metric))
                        return

            # If we are here, then there was no convergence. We therefore need to train on the training data (again). We
            # first shuffle the training data so that we aren't learning on the exact same mini batches as last time.
            random.shuffle(numpy_training_data)
            # Now we form the mini batches. Each mini batch is a list of examples.
            mini_batches = [numpy_training_data[k:k + mini_batch_size] for k in range(0, len(numpy_training_data), mini_batch_size)]
            # We now perform gradient descent on each mini batch. We also maintain the delta weights from the previous
            # mini batch so that we can apply momentum to our current delta weight.
            prev_delta_weights = None
            for mini_batch in mini_batches:
                prev_delta_weights = self.train_mini_batch(mini_batch, prev_delta_weights)

    def train_mini_batch(self, mini_batch, prev_weights):
        total_dw = [np.zeros(w.shape) for w in self.weights]
        for example_array, expected_class in mini_batch:
            expected_array = self.get_class_array(expected_class)
            delta_weights = self.back_propagation(example_array, expected_array)
            for i in range(len(self.weights)):
                delta_weights[i] *= (self.learning_rate / len(mini_batch))
                if prev_weights is not None:
                    delta_weights[i] -= self.momentum * prev_weights[i]
                self.weights[i] -= delta_weights[i]
                total_dw[i] += delta_weights[i]
        return total_dw

    def get_error(self, data_set):
        numpy_data = data_set.get_numpy_list()
        squared_sum = 0
        for example_array, expected_class in numpy_data:
            output = self.run(example_array)
            squared_sum += (self.get_class_value(output) - expected_class)**2
        return math.sqrt(squared_sum) / len(numpy_data)

    def get_accuracy(self, data_set):
        numpy_data = data_set.get_numpy_list()
        correct = 0
        for example_array, expected_class in numpy_data:
            output = self.run(example_array)
            if self.get_class_value(output) == expected_class:
                correct += 1
        return correct / len(numpy_data)

    def init_weights(self):
        weights = []
        for size in range(1, len(self.layer_size)):
            weights.append(np.random.randn(self.layer_size[size], self.layer_size[size-1] + 1))
        return weights

    def get_activation(self, example: np.ndarray):
        inputs = np.append(example, af.sigmoid(1))
        activation = [inputs]
        for i in range(len(self.weights)):
            inputs = np.dot(self.weights[i], inputs)
            if not (self.is_regression() and i == len(self.weights) - 1):
               inputs = af.sigmoid(inputs)
            if i < len(self.weights) - 1:
                inputs = np.append(inputs, af.sigmoid(1))
            activation.append(inputs)
        return activation

    def back_propagation(self, example: np.ndarray, expected: np.ndarray):
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        n = len(self.layer_size)

        # Feedforward
        activation = self.get_activation(example)

        # Output Layer
        delta = af.cost_prime(expected, activation[-1])
        if not self.is_regression():
            delta = delta * af.sigmoid_prime(activation[-1])
        delta_weights[-1] = np.outer(delta, activation[-2])
        # Hidden Layer
        for i in range(n-2, 0, -1):
            previous_activation = activation[i-1]
            downstream_weights = self.weights[i].T
            if i < n-2:
                delta = delta[:-1]
            delta = np.dot(downstream_weights, delta) * af.sigmoid_prime(activation[i])
            delta_weights[i-1] = np.delete(np.outer(delta, previous_activation), -1, 0)
        return delta_weights

    def run(self, example):
        return self.get_activation(example)[-1]


