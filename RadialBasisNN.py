import math
class RBFNN:
    def __init__(self, centers, validation_data, inputs, outputs, learning_rate):
        self.receptors = centers
        self.validation_data = validation_data
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate

    def runRBFNN(self):
        # First run gradient descent to tune weights
        # Send an input through each
        # Proven to be a universal approximator, function approximation, interpolation,
        # classification, time-series prediction

        #

    def rbMethod(self):
        # Activation Function -- Throw this function into activation functions

        # Find variance: Squared sum of distances between the respective receptor
        # and each clusters nearest sample. Inputs trial and error,
        input
        # hidden nodes is first ex 4
        # input nodes is second for matrix ex 3
        # 4x3
        for i in range(len(self.inputs)-1):
            math.e**((-math.abs(inputs[i] - center[i])**2)/(variance**2))


