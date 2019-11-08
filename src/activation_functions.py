# activation_functions.py
# This file stores the different activations functions and their derivatives that we either used or considered using in
# our networks.

import numpy as np

# Regular sigmoid function, using logistic function.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Derivative of logistic function.
def sigmoid_prime(z):
    return z*(1-z)


# Hyperbolic tangent.
def tanh(z):
    return (2 / 1 + np.exp(-2*z)) - 1


# Derivative of hyperbolic tangent.
def tanh_prime(z):
    return 1 - np.power(tanh(z), 2)


# Squared error derivative.
def cost_prime(expected, actual):
    return actual - expected


# ReLu function.
def relu(z):
    return np.maximum(0, z)


# Softmax function.
def softmax(self, z):
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator/denominator