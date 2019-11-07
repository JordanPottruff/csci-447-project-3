import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return z*(1-z)


def tanh(z):
    return (2 / 1 + np.exp(-2*z)) - 1


def tanh_prime(z):
    return 1 - np.power(tanh(z), 2)


def cost_prime(expected, actual):
    return actual - expected

def relu(z):
    return np.maximum(0, z)

    # one alternative to reLU is the softplus which is an approximation of reLU
    # def softplus(self, z):
    #     return np.log(1.0 + np.exp(z))

    # def softplus_prime(self, z):
    #     return 1 / (1 + np.exp(-z))
