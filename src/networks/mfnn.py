# mfnn.py
# Defines the mutli-layer feedforward neural network implementation.

import numpy as np
import random
import math
import src.activation_functions as af

# The amount each new average metric needs to be better than the old average metric for the training process to
# continue.
CONVERGENCE_THRESHOLD = .0001


# The class for creating MFNN objects.
class MFNN:

    # Creates an MFNN out of:
    # - training_data: DataSet object for training data.
    # - validation_data: DataSet object for validation data.
    # - layer_size: list of layer sizes, including input at start, output at end, and hidden layer sizes in between.
    # - learning_rate: float specifying the learning rate to be used.
    # - momentum: float specifying how much momentum to be used.
    # - convergence_size: how many previous epochs to evaluate when determining convergence.
    # - classes: the class values for the data, if a classification problem.
    def __init__(self, training_data, validation_data, layer_size, learning_rate, momentum, convergence_size, classes=None):
        self.training_data = training_data
        self.validation_data = validation_data
        self.num_layers = len(layer_size)
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.convergence_size = convergence_size
        # We initialize the weights randomly.
        self.weights = self.init_weights()
        # The class dictionary is used to map each class value to an index in the output array of the network.
        self.class_dict = None if classes is None else {cls: index for index, cls in enumerate(classes)}

    # Returns true if this network is running on a regression data set.
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

    # Trains the network on the training data, ending when the accuracy/error has converged on the validation data.
    def train(self):
        # We convert our training data to a list of tuples that stores the numpy array of the example and the class.
        numpy_training_data = self.training_data.get_numpy_list()
        # We always use mini batches with four examples in it.
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
                print(new_metric)
                if self.is_regression():
                    # Error needs to invert the difference, as we are MINIMIZING error.
                    if -difference < CONVERGENCE_THRESHOLD:
                        return
                else:
                    # We attempt to MAXIMIZE accuracy for classification data.
                    if difference < CONVERGENCE_THRESHOLD:
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

    # This trains on a single mini batch. It takes in the mini batch being trained on as well as the previous delta
    # weights from the last mini batch, or None if this is the first mini batch.
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

    # Returns the root mean squared error on the specified data set according to the current configuration of weights
    # in the network.
    def get_error(self, data_set):
        numpy_data = data_set.get_numpy_list()
        squared_sum = 0
        # Sum up the squared sup across all squared differences between the actual class value and the expected value.
        for example_array, expected_class in numpy_data:
            output = self.run(example_array)
            squared_sum += (self.get_class_value(output) - expected_class)**2
        return math.sqrt(squared_sum) / len(numpy_data)

    # Returns the accuracy on the specified data set according to the current configuration of weights in the network.
    def get_accuracy(self, data_set):
        numpy_data = data_set.get_numpy_list()
        correct = 0
        # Sum the number of correctly classified examples.
        for example_array, expected_class in numpy_data:
            output = self.run(example_array)
            if self.get_class_value(output) == expected_class:
                correct += 1
        # Divide the number of correct examples by the total number of examples.
        return correct / len(numpy_data)

    # Initialize the weights to random values that are generated according to a guassian distribution centered at zero
    # with a standard deviation of one.
    def init_weights(self):
        weights = []
        for size in range(1, len(self.layer_size)):
            weights.append(np.random.randn(self.layer_size[size], self.layer_size[size-1] + 1))
        return weights

    # Returns a list of numpy arrays that are the activations of each layer caused by a given example inputted into the
    # network. The first activation is just the example (plus a bias activation) and the last activation is the output
    # of the network.
    def get_activation(self, example: np.ndarray):
        inputs = np.append(example, af.sigmoid(1))
        activation = [inputs]
        # We successively apply each weight matrix to the example using the dot product, and then store that as the
        # activation that should be dotted with the next weight matrix, and so on.
        for i in range(len(self.weights)):
            # We now update input to be the activation of the next layer.
            inputs = np.dot(self.weights[i], inputs)
            # We do not want to apply the sigmoid function if this is the output layer of a regression problem.
            if not (self.is_regression() and i == len(self.weights) - 1):
               inputs = af.sigmoid(inputs)
            # We also do not want to append the activation of the bias node (sigmoid(1)) to the activation if this is
            # the output layer.
            if i < len(self.weights) - 1:
                inputs = np.append(inputs, af.sigmoid(1))
            # We then add the current activation to the list of activations.
            activation.append(inputs)
        # Activation is now a list of numpy arrays, each an activation of a layer.
        return activation

    # Performs back propagation of a specified example given an expected output.
    def back_propagation(self, example: np.ndarray, expected: np.ndarray):
        # Objective: calculate the change in weights (delta_weights) for the gradient formed by this training example.
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        n = len(self.layer_size)

        # First we perform feedforward, and save all the activations of each layer.
        activation = self.get_activation(example)

        # We then calculate the delta of the output layer.
        delta = af.cost_prime(expected, activation[-1])
        if not self.is_regression():
            delta = delta * af.sigmoid_prime(activation[-1])
        # The change in weights for the output layer is then updated using delta.
        delta_weights[-1] = np.outer(delta, activation[-2])

        # Now, we calculate the delta and change in weights for each hidden layer weights. The variable i represents the
        # layer we are currently evaluating, with i=1 being the first hidden layer, i=2 being the second, etc.
        for i in range(n-2, 0, -1):
            # This is the activation from the previous hidden (or the input) layer.
            previous_activation = activation[i-1]
            # These are the weights going out from the current layer.
            downstream_weights = self.weights[i].T

            # We need to trim the delta from the previous layer (unless it comes from the output layer) because it
            # includes a delta for the bias node, which does not change.
            if i < n-2:
                delta = delta[:-1]
            # We compute delta for this layer:
            delta = np.dot(downstream_weights, delta) * af.sigmoid_prime(activation[i])
            # Then we update the delta_weight matrix to have the change in weights for this layer's weights:
            delta_weights[i-1] = np.delete(np.outer(delta, previous_activation), -1, 0)
        # We return delta_weights, which is a list of the matrices to be subtracted from the actual weights.
        return delta_weights

    # This method returns the output activation layer of a single input example.
    def run(self, example):
        return self.get_activation(example)[-1]


