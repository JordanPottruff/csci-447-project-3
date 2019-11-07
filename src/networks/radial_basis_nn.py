import math
import numpy as np
from src.data import data_set as d
from src import activation_functions as af

class RBFNN:

    def __init__(self, centers, training_data, validation_data, num_inputs, classes, learning_rate, mini_batching=32):
        self.receptors = centers  # Dataset, use for rbf (Centers of Gaussians)
        self.training_data = training_data  # Dataset, use to train
        self.validation_data = validation_data # Dataset, use to validate training convergence
        self.num_inputs = num_inputs
        self.classes = classes  # list of class outputs
        self.num_outputs = len(self.classes) if self.classes is not None else 1
        self.learning_rate = learning_rate
        self.weights = np.random.randn(self.num_outputs, len(self.receptors.data))
        self.mini_batching = mini_batching
        self.epochs = 1
        self.batch_size = math.ceil(len(training_data.data)/10)
        self.std_dev = self.get_stdrd_dev()
        self.class_dict = None if classes is None else {cls: index for index, cls in enumerate(classes)}

    # Returns the value of the class with the highest activation in the given class array. See 'get_class_array' for a
    # better understanding of the purpose of these functions.
    # def get_class_value(self, class_array):
    #     if self.is_regression():
    #         return class_array[0]
    #     else:
    #         inverted_class_dict = {value: key for key, value in self.class_dict.items()}
    #         max_index = 0
    #         for i in range(1, len(class_array)):
    #             if class_array[i] > class_array[max_index]:
    #                 max_index = i
    #         return inverted_class_dict[max_index]

    def get_stdrd_dev(self):
        """Get the standard deviation between the clusters of the center nodes"""
        max_dist_bw_clusters = self.receptors.get_max_distance()
        print(max_dist_bw_clusters)
        num_cluster_centers = len(self.receptors.data)
        stdrd_dev = max_dist_bw_clusters/math.sqrt((2*num_cluster_centers))
        return stdrd_dev

    def get_rbf_activation(self, input, center, stdrd_dev):
        """Use a Gaussian RBF as our "activation" function and take in an input parameter and centers for the neural
         network and the standard deviation for these centers. Then calculate and output the 'activation value'."""
        activation_value = math.exp(-1*(self.receptors.distance(input, center))**2/(2*stdrd_dev**2))
        return activation_value

    def run_rbfnn(self, example: np.ndarray):
        """Uses a linear combination of Gaussians to approximate any function."""
        stdrd_dev = self.std_dev
        num_of_centers = len(self.receptors.data)
        bias = 1
        hidden_activations = []
        for idx in range(num_of_centers):
            gaussian_rbf = self.get_rbf_activation(example, self.receptors.data[idx], stdrd_dev)
            hidden_activations.append(gaussian_rbf)
        hidden_activations = np.array(hidden_activations)
        output_activations = np.dot(self.weights, hidden_activations)
        if not self.is_regression():
            output_activations = af.sigmoid(output_activations)
        return output_activations, hidden_activations

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
        # The following method call creates a new list that stores tuples. The first position of each tuple is the
        # example/observation, in the form of a numpy array. The second position is the name of the class.
        numpy_training_data = self.training_data.get_numpy_list()
        for epoch in range(self.epochs):
            # Now we form the mini batches. Each mini batch is split of our training data. These will be used to
            # calculate model error and update model coefficients. So first split training set into batches
            mini_batches = [numpy_training_data[k:k + self.mini_batching] for k in
                            range(0, len(numpy_training_data), self.mini_batching)]
            mini_batch_delta_weights = []
            for example, expected_class in mini_batches:
                    # We now perform gradient descent on each mini batch and get the delta-weights
                    expected_output = self.get_class_array(expected_class)
                    delta_weights = self.gradient_descent(example, expected_class)
                    # TODO: potentially use an average delta weight
                    self.weights -= delta_weights
                    mini_batch_delta_weights.append(delta_weights)
                    print(self.weights)


def test():
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
    rbfnn = RBFNN(centers, training, validation, 2, None, 1)
    rbfnn.run_rbfnn(np.array([0, 1]))
    # Train the rbfnn on its training data.
    rbfnn.train()
    # Next, we would run a bunch of test examples.

test()
