import numpy as np
import src.data.data as data


class MFNN:
    def __init__(self, training_data, validation_data, layer_size, learning_rate, momentum, regression):
        self.training_data = training_data
        self.validation_data = validation_data
        self.num_layers = len(layer_size)
        self.layer_size = layer_size
        self.weights = self.init_weights()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regression = regression

    def train(self):
        mini_batch_size = 4
        for i in range(1000):
            self.training_data
            mini_batches = [self.training_data.get_data()[k:k + mini_batch_size] for k in range(0, len(self.training_data.get_data()), mini_batch_size)]
            for mini_batch in mini_batches:
                self.train(mini_batch)




    def init_weights(self):
        weights = []
        for size in range(1, len(self.layer_size)):
            weights.append(np.random.randn(self.layer_size[size], self.layer_size[size-1] + 1))
        return weights

    def get_activation(self, example: list):
        inputs = np.append(np.array(example), sigmoid(1))
        activation = [inputs]
        for i in range(len(self.weights)):
            inputs = sigmoid(np.dot(self.weights[i], inputs))
            if i < len(self.weights) - 1:
                inputs = np.append(inputs, sigmoid(1))
            activation.append(inputs)
        return activation

    def backpropagation(self, example: list, expected: list):
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        n = len(self.layer_size)

        # Feedforward
        activation = self.get_activation(example)

        # Output Layer
        delta = cost_prime(expected, activation[-1]) * sigmoid_prime(activation[-1])
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

    def training_set(self, mini_batch):
        for example, expected in mini_batch:
            delta_weights = self.backpropagation(example, expected)
            self.weights = [w-(self.learning_rate/len(mini_batch))*dw for w, dw in zip(self.weights, delta_weights)]


def sigmoid_prime(sigmoid):
    return sigmoid*(1-sigmoid)


def cost_prime(expected, actual):
    return actual - expected


def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-x))





















