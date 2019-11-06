import math

# hidden nodes is first ex 4
# input nodes is second for matrix ex 3
# 4x3
        
class RBFNN:
    def __init__(self, centers, validation_data, inputs, outputs, learning_rate):
        self.receptors = centers
        self.validation_data = validation_data
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
    
    def get_stdrd_dev(self, centers):
        # I will throw this function into k-means, as we will only use this when doing the k-means... maybe
        # more... as I can't find another one to use but maybe we can use a similar method for CKNN or EKNN

        # If we use k-means clustering to select prototypes, then one simple method
        # for specifying the Beta coefficients is to set sigma equal to the average
        # distance between all points in the cluster and the cluster center
        cluster_center = 0
        sum_of_distances = 0
        for point in k_means_set:
            sum_of_distances = math.abs(point - cluster_center)
        return sum_of_distances/total_pnts_in_k_means_set
    
    def training(self, validation_data):
        pass
    def runRBFNN(self):
        # First run gradient descent to tune weights
        # Send an input through each
        # Proven to be a universal approximator, function approximation, interpolation,
        # classification, time-series prediction
        
    def activate_with_RBF(self, input, center, stdrd_dev):
        # Activation Function -- Throw this function into activation functions
        """Uses a linear combination of Gaussians to approximate any function. To run this function we need 
        where to place the centers and their standard deviations"""
        # Standard Deviation: 1. set to be that of the points assigned to a particular cluster
        # 2. Or we could use a standard deviation for all clusters => maximum distance b/w any two cluster centers divided 
        # by squrt(2 * number of clusters)
        return  math.e**((-math.abs(input[i] - center[i])**2)/(2*(stdrd_dev**2)))
           


