import numpy as np

class Neuron(object):

    """
    Simple class for storing each neuron's data (weights and bias etc.) in the neural network.
    """

    def __init__(self,i,w,b):
        self.inputNeurons = i   # list of neurons whose values feed into this neuron
        self.inputValues = None
        self.weights = np.array(w) # list of the corresponting weights
        self.bias = b
        self.outputValue = None

    def getInputValues(self):
        self.inputValues = []
        for inputNeuron in self.inputNeurons:
            self.inputValues.append(inputNeuron.outputValue)
        self.inputValues = np.array(self.inputValues)

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def getOutputValue(self):
        self.outputValue = self.sigmoid(np.dot(self.weights, self.inputValues) + self.bias)
