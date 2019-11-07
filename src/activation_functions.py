import numpy as np


class ActivationFunctions:

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def tanh(self, z):
        return ((2 / 1 + np.exp(-2*z)) - 1)

    def tanh_prime(self, z):
        return 1 - np.power(self.tanh(z), 2)

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        return numerator/denominator

    # one alternative to reLU is the softplus which is an approximation of reLU
    # def softplus(self, z):
    #     return np.log(1.0 + np.exp(z))

    # def softplus_prime(self, z):
    #     return 1 / (1 + np.exp(-z))
