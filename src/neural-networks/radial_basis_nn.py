import math
import numpy as np
from ..src.data import data as d
from ..src import activation_functions as af
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
        self.weights = np.random.randn(len(self.receptors.data))  #TODO is this correct
        self.epochs = 100

    def get_stdrd_dev(self):
        #  I need max distance between any 2 cluster centers and divide by sqrt(2*number of cluster centers)
        # Find distances between centroids and take max distance
        # Find max distance between centers from eknn
        max_dist_bw_clusters = self.receptors.get_stdrd_dev()
        num_cluster_centers = len(self.receptors.data)
        stdrd_dev = max_dist_bw_clusters/math.sqrt((2*num_cluster_centers))
        return stdrd_dev

    def training(self, validation_data):
        pass

    def get_rbf_activation(self, input, center, stdrd_dev):
        activation_value = math.exp(-1*(self.receptors.distance(input, center))**2/(2*stdrd_dev))
        return activation_value

    def run_rbfnn(self):
        """Uses a linear combination of Gaussians to approximate any function."""
        stdrd_dev = self.get_stdrd_dev()
        print(stdrd_dev)
        num_of_centers = len(self.receptors.data)
        print(num_of_centers)
        bias = 1
        # Weights and bias defined in class
        predictions = []
        for idx in range(num_of_centers):
            gaussian_rbf = self.get_rbf_activation(self.inputs.data[idx], self.receptors.data[idx], stdrd_dev)
            predictions.append(self.weights * gaussian_rbf + bias) #TODO Need to verify this is correct and train
            
        return predictions
