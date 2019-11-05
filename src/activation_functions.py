import numpy as np
import math

class ActivationFunctions:

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def tanh(self, z):
        return ((2 / 1 + np.exp(-2*z)) - 1)

    def tanh_prime(self, z):
        return 1 - np.power(self.tanh(z), 2)

