import math
import numpy as np
from Project_3.src.data import data as d
from src import activation_functions as af
# hidden nodes is first ex 4
# input nodes is second for matrix ex 3
# 4x3
        
class RBFNN:
    def __init__(self, centers, training_data, test_data, outputs, learning_rate):
        self.receptors = centers  # Dataset, use for rbf (Centers of Gaussians)
        self.training_data = training_data  # Dataset, use to train
        self.inputs = test_data  # Dataset, use to test
        self.outputs = outputs  # list of class outputs
        self.learning_rate = learning_rate
        self.weights = np.random.randn(len(self.outputs), len(self.receptors.data))
        self.b = np.random.rand(1)
        self.epochs = 100
        self.batch_size = len(training_data.data)/10

    def get_stdrd_dev(self):
        """Get the standard deviation between the clusters of the center nodes"""
        max_dist_bw_clusters = self.receptors.get_max_distance_clusters()
        num_cluster_centers = len(self.receptors.data)
        stdrd_dev = max_dist_bw_clusters/math.sqrt((2*num_cluster_centers))
        return stdrd_dev

    def training(self):
        for epoch in range(self.epochs):
            for i in range(self.batch_size):


    def get_rbf_activation(self, input, center, stdrd_dev):
        """Use a Gaussian RBF as our "activation" function and take in an input parameter and centers for the neural
         network and the standard deviation for these centers. Then calculate and output the 'activation value'."""
        activation_value = math.exp(-1*(self.receptors.distance(input, center))**2/(2*stdrd_dev**2))
        return activation_value

    
    def run_rbfnn(self):
        """Uses a linear combination of Gaussians to approximate any function."""
        stdrd_dev = self.get_stdrd_dev()
        print(stdrd_dev)
        num_of_centers = len(self.receptors.data)
        print(num_of_centers)
        bias = 1
        # Weights and bias defined in class ignore bias for now
        for epoch in range(self.epochs):
            activations= []
            for idx in range(num_of_centers):
                gaussian_rbf = self.get_rbf_activation(self.inputs.data[idx], self.receptors.data[idx], stdrd_dev)
                activations.append(gaussian_rbf)
                # Weights time the output of activation function
            
            F = np.dot(self.weights, np.dot(self.weights, activations))
            # Calculate Error
            # Output Layer
            
            
            self.weights = self.weights - self.learning_rate * activations * error
            
        #return predictions
