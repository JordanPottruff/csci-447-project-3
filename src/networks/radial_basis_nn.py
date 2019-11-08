import math
import numpy as np
from src.data import data_set as d
from src import activation_functions as af
import random

# The amount each new average metric needs to be better than the old average metric for the training process to
# continue.
CONVERGENCE_THRESHOLD = .0001


class RBFNN:

    def __init__(self, centers, training_data, validation_data, num_inputs, learning_rate, convergence_size, classes=None):
        self.receptors = centers  # Dataset, use for rbf (Centers of Gaussians)
        self.training_data = training_data  # Dataset, use to train
        self.validation_data = validation_data # Dataset, use to validate training convergence
        self.num_inputs = num_inputs
        self.classes = classes  # list of class outputs
        self.num_outputs = len(self.classes) if self.classes is not None else 1
        self.learning_rate = learning_rate
        self.convergence_size = convergence_size
        self.weights = np.random.randn(self.num_outputs, len(self.receptors.data))
        self.epochs = 1
        self.batch_size = math.ceil(len(training_data.data)/10)
        self.std_dev = self.get_stdrd_dev()
        self.class_dict = None if classes is None else {cls: index for index, cls in enumerate(classes)}
        self.dist_cache = {}
        self.counter = 0

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

    def get_stdrd_dev(self):
        """Get the standard deviation between the clusters of the center nodes"""
        max_dist_bw_clusters = self.receptors.get_max_distance()
        num_cluster_centers = len(self.receptors.data)
        stdrd_dev = max_dist_bw_clusters/math.sqrt((2*num_cluster_centers))
        return stdrd_dev

    def get_rbf_activation(self, input, center, stdrd_dev):
        """Use a Gaussian RBF as our "activation" function and take in an input parameter and centers for the neural
         network and the standard deviation for these centers. Then calculate and output the 'activation value'."""
        if center not in self.dist_cache:
            self.dist_cache[center] = {}
        if input not in self.dist_cache[center]:
            self.dist_cache[center][input] = self.receptors.distance(input, center)
        dist = self.dist_cache[center][input]
        activation_value = math.exp((-1*(dist)**2)/(2*stdrd_dev**2))
        return activation_value

    def print_weights(self):
        for weight_i in range(len(self.weights)):
            self.print_weight(weight_i)

    def print_weight(self, i):
        print(np.array2string(self.weights[i], precision=2, sign='+', separator=', ', suppress_small=True))
        print()

    def print_centers(self):
        for center in self.receptors.data:
            rounded_center = []
            for col in self.receptors.attr_cols:
                rounded_center.append(round(center[col]*10000)/10000)
            print(rounded_center)

    def run_rbfnn(self, example: list):
        self.counter += 1
        """Uses a linear combination of Gaussians to approximate any function."""
        stdrd_dev = self.std_dev
        num_of_centers = len(self.receptors.data)
        hidden_activations = []
        min_dist = float("inf")
        min_receptor = None
        max_dist = float("-inf")
        max_receptor = None
        for idx in range(num_of_centers):
            gaussian_rbf = self.get_rbf_activation(example, self.receptors.data[idx], stdrd_dev)
            if min_receptor is None or min_dist > gaussian_rbf:
                min_receptor = self.receptors.data[idx]
                min_dist = gaussian_rbf
            if max_receptor is None or max_dist < gaussian_rbf:
                max_receptor = self.receptors.data[idx]
                max_dist = gaussian_rbf
            hidden_activations.append(gaussian_rbf)

        hidden_activations = np.array(hidden_activations)
        output_activations = np.dot(self.weights, hidden_activations)
        if not self.is_regression():
            output_activations = af.sigmoid(output_activations)
        return output_activations, hidden_activations

    def round_example(self, example):
        rounded_example = []
        for col in self.training_data.attr_cols:
            rounded_example.append(round(example[col]*1000)/1000)
        return rounded_example

    def gradient_descent(self, example: np.ndarray, expected: np.ndarray):
        output_activation, hidden_activations = self.run_rbfnn(example)
        delta = af.cost_prime(expected, output_activation)
        # No sigmoid activation function is we are doing regression, so we don't need to use the derivative of the
        # sigmoid during gradient descent in that case.
        if not self.is_regression():
            delta = delta * af.sigmoid_prime(output_activation)
        delta_weights = np.outer(delta, hidden_activations)
        return delta_weights

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

    def train(self):
        training_observations = []
        for example in self.training_data.data:
            training_observations.append((example, example[self.training_data.class_col]))

        mini_batch_size = 8
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
            if count == 1:
                print("Initial Error: " + "{:.2f}".format(metric))
            # Wait until the convergence check list has all self.convergence_size*2 items.
            if count % 5 == 0:
                print("Error so far... " + "{:.2f}".format(metric))
                print("Weights so far...")
                self.print_weights()
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

                if self.is_regression():
                    # Error needs to invert the difference, as we are MINIMIZING error.
                    if -difference < CONVERGENCE_THRESHOLD:
                        print("Final Validation Error: " + "{:.2f}".format(metric))
                        print("Final Weights")
                        return
                else:
                    # We attempt to MAXIMIZE accuracy for classification data.
                    if difference < CONVERGENCE_THRESHOLD:
                        print("Final Validation Accuracy: " + "{:.2f}%".format(metric))
                        print("Final Weights")
                        return

            # If we are here, then there was no convergence. We therefore need to train on the training data (again). We
            # first shuffle the training data so that we aren't learning on the exact same mini batches as last time.
            random.shuffle(training_observations)
            # Now we form the mini batches. Each mini batch is a list of examples.
            mini_batches = [training_observations[k:k + mini_batch_size] for k in range(0, len(training_observations), mini_batch_size)]
            # We now perform gradient descent on each mini batch. We also maintain the delta weights from the previous
            # mini batch so that we can apply momentum to our current delta weight.
            prev_delta_weights = None
            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch)

    def train_mini_batch(self, mini_batch):
        for example_array, expected_class in mini_batch:
            expected_array = self.get_class_array(expected_class)
            delta_weights = self.gradient_descent(example_array, expected_array) * (self.learning_rate / len(mini_batch))
            self.weights -= delta_weights

    def get_error(self, data_set):
        observations = [(example, example[self.training_data.class_col]) for example in data_set.data]
        squared_sum = 0
        for example, expected_class in observations:
            output = self.run(example)
            squared_sum += (self.get_class_value(output) - expected_class)**2
        return math.sqrt(squared_sum) / len(observations)

    def get_accuracy(self, data_set):
        observations = [(example, example[self.training_data.class_col]) for example in data_set.data]
        correct = 0
        for example, expected_class in observations:
            output = self.run(example)
            if self.get_class_value(output) == expected_class:
                correct += 1
        return correct / len(observations)

    def get_numpy_array(self, example):
        attr_only = []
        for col in self.training_data.attr_cols:
            attr_only.append(example[col])
        return np.array(attr_only), example[self.training_data.class_col]

    def run(self, example):
        out, hidden = self.run_rbfnn(example)
        return out


def test_regression():
    # This is just to setup some testing data.
    data = d.get_regression_test_data("../../data/test/regression_test_set.data")
    test = data.copy()
    training = data.copy()
    training.data *= 10
    validation = data.copy()
    validation.data *= 3

    # I'm manually writing out the centers here.
    centers_data = [[0, 1, 10], [3, 1, 13], [2, 2, 14], [3, 3, 19], [3, 4, 22]]
    centers = data.copy()
    centers.data = centers_data

    # Create the network
    rbfnn = RBFNN(centers, training, validation, 2, 1, 100)
    # Train the rbfnn on its training data.
    rbfnn.train()
    # Next, we would run a bunch of test examples.


def test_classification():
    # This is just to setup some testing data.
    data = d.get_classification_test_data("../../data/test/classification_test_set.data")
    test = data.copy()
    training = data.copy()
    training.data *= 10
    validation = data.copy()
    validation.data *= 3

    # I'm manually writing out the centers here.
    centers_data = [[0, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 0]]
    centers = data.copy()
    centers.data = centers_data

    # Create the network
    rbfnn = RBFNN(centers, training, validation, 2, 1, 100, [0, 1])
    # Train the rbfnn on its training data.
    rbfnn.train()
    # Next, we would run a bunch of test examples.


# test_classification()