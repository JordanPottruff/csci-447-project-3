# rbfnn.py
# Defines the radial basis network.

import math
import numpy as np
from src import activation_functions as af
import random

# The amount each new average metric needs to be better than the old average metric for the training process to
# continue.
CONVERGENCE_THRESHOLD = .0001


# Creates an instance of an RBFNN.
class RBFNN:

    # Creates an RBFNN using:
    # - centers: a DataSet object that contains the center values for our radial basis (hidden) nodes.
    # - training_data: a DataSet object that contains training examples.
    # - validation_data: a DataSet object for validation examples.
    # - num_inputs: the size of the input layer.
    # - learning_rate: the learning rate for the algorithm.
    # - convergence_size: how many epochs to evalaute when determing whether the algorithm has converged.
    # - classes: the class values for the data, if a classification problem.
    def __init__(self, centers, training_data, validation_data, num_inputs, learning_rate, convergence_size, classes=None):
        self.receptors = centers  # Dataset, use for rbf (Centers of Gaussians)
        self.training_data = training_data  # Dataset, use to train
        self.validation_data = validation_data # Dataset, use to validate training convergence
        self.num_inputs = num_inputs
        self.num_outputs = len(classes) if classes is not None else 1
        self.learning_rate = learning_rate
        self.convergence_size = convergence_size
        self.weights = np.random.randn(self.num_outputs, len(self.receptors.data))
        # Calculates the standard deviation of our training data.
        self.std_dev = self.get_stdrd_dev()
        # Creates a mapping of each class value to an index in the output array.
        self.class_dict = None if classes is None else {cls: index for index, cls in enumerate(classes)}
        # To speed up the algorithm, we cache the distances.
        self.dist_cache = {}

    # Returns the value of the class with the highest activation in the given class array. See 'get_class_array' for a
    # better understanding of the purpose of these functions.
    def get_class_value(self, class_array):
        if self.is_regression():
            return class_array[0]
        else:
            # Flip class dictionary to go from output array index to associated class value.
            inverted_class_dict = {value: key for key, value in self.class_dict.items()}
            # Find the max index.
            max_index = 0
            for i in range(1, len(class_array)):
                if class_array[i] > class_array[max_index]:
                    max_index = i
            # Return the class associated with that max index.
            return inverted_class_dict[max_index]

    # Gets the estimated standard deviation between the center vectors.
    def get_stdrd_dev(self):
        max_dist_bw_clusters = self.receptors.get_max_distance()
        num_cluster_centers = len(self.receptors.data)
        stdrd_dev = max_dist_bw_clusters/math.sqrt((2*num_cluster_centers))
        return stdrd_dev

    # Gets the activations of the radial basis layer of the network for a specific input and center.
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

    # Gets the output and hidden layer activations for a specific example inputted to the network.
    def run_rbfnn(self, example: list):
        """Uses a linear combination of Gaussians to approximate any function."""
        stdrd_dev = self.std_dev
        num_of_centers = len(self.receptors.data)

        # Find the activations of the hidden layer.
        hidden_activations = []
        for idx in range(num_of_centers):
            gaussian_rbf = self.get_rbf_activation(example, self.receptors.data[idx], stdrd_dev)
            hidden_activations.append(gaussian_rbf)
        hidden_activations = np.array(hidden_activations)

        # Find the activations of the output layer.
        output_activations = np.dot(self.weights, hidden_activations)
        if not self.is_regression():
            output_activations = af.sigmoid(output_activations)

        # Return the activations of both layers.
        return output_activations, hidden_activations

    # Performs gradient descent on our single weight matrix connecting the radial basis (hidden) layer and output layer.
    # Does gradient descent on a single example and its expected output, returns the change in weights calculated from
    # the gradient.
    def gradient_descent(self, example: np.ndarray, expected: np.ndarray):
        # Get activations for example.
        output_activation, hidden_activations = self.run_rbfnn(example)
        delta = af.cost_prime(expected, output_activation)
        # No sigmoid activation function is we are doing regression, so we don't need to use the derivative of the
        # sigmoid during gradient descent in that case.
        if not self.is_regression():
            delta = delta * af.sigmoid_prime(output_activation)
        # Calculates the delta weight given the delta.
        delta_weights = np.outer(delta, hidden_activations)
        return delta_weights

    # Returns true if the current data set is a regression problem.
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

    # Trains the data on the training data provided to the algorithm. Does so using mini batches of size four.
    def train(self):
        # We want both the original example and the expected class of the example.
        training_observations = []
        for example in self.training_data.data:
            training_observations.append((example, example[self.training_data.class_col]))
        # We always use mini batches with four examples ini t.
        mini_batch_size = 4
        # This will record the error/accuracy of the last 'self.convergence_size' epochs. We can then evaluate how
        # the older and newer values in this list compare to determine if convergence has occurred.
        convergence_check = []

        # Each cycle of this while loop is a single epoch. That is, it covers all of the training data (shuffled in
        # random order and put into mini batches). This loop will break based on the convergence calculation used
        # immediately below.
        while True:
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
                if self.is_regression():
                    # Error needs to invert the difference, as we are MINIMIZING error.
                    if -difference < CONVERGENCE_THRESHOLD or metric < .1:
                        return
                else:
                    # We attempt to MAXIMIZE accuracy for classification data.
                    if difference < CONVERGENCE_THRESHOLD:
                        return

            # If we are here, then there was no convergence. We therefore need to train on the training data (again). We
            # first shuffle the training data so that we aren't learning on the exact same mini batches as last time.
            random.shuffle(training_observations)
            # Now we form the mini batches. Each mini batch is a list of examples.
            mini_batches = [training_observations[k:k + mini_batch_size] for k in range(0, len(training_observations), mini_batch_size)]
            # We now perform gradient descent on each mini batch. We also maintain the delta weights from the previous
            # mini batch so that we can apply momentum to our current delta weight.
            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch)

    # This trains on a single mini batch.
    def train_mini_batch(self, mini_batch):
        for example_array, expected_class in mini_batch:
            expected_array = self.get_class_array(expected_class)
            # Calculate teh delta weights as the gradient descent output times the learning rate divided by the number
            # of items in the mini batch (averaging).
            delta_weights = self.gradient_descent(example_array, expected_array) * (self.learning_rate / len(mini_batch))
            # Applies the change in weights for this example in this mini batch.
            self.weights -= delta_weights

    # Returns the root mean squared error on the specified data set according to the current configuration of weights
    # in the network.
    def get_error(self, data_set):
        observations = [(example, example[self.training_data.class_col]) for example in data_set.data]
        squared_sum = 0
        # Sum up the squared sup across all squared differences between the actual class value and the expected value.
        for example, expected_class in observations:
            output = self.run(example)
            squared_sum += (self.get_class_value(output) - expected_class)**2
        return math.sqrt(squared_sum) / len(observations)

    # Returns the accuracy on the specified data set according to the current configuration of weights in the network.
    def get_accuracy(self, data_set):
        observations = [(example, example[self.training_data.class_col]) for example in data_set.data]
        correct = 0
        # Sum the number of correctly classified examples.
        for example, expected_class in observations:
            output = self.run(example)
            if self.get_class_value(output) == expected_class:
                correct += 1
        # Divide the number of correct examples by the total number of examples.
        return correct / len(observations)

    def get_numpy_array(self, example):
        attr_only = []
        for col in self.training_data.attr_cols:
            attr_only.append(example[col])
        return np.array(attr_only), example[self.training_data.class_col]

    # This method returns the output activation layer of a single input example.
    def run(self, example):
        out, hidden = self.run_rbfnn(example)
        return out
