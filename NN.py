import numpy as np
import random
from Neuron import Neuron
from NN_Display import NN_Display

class NN(object):

    """
    Class for defining and training neural networks using stochastic gradient descent.
    One defines a network structure with the input dimension, and then a list [x,...] giving the number of neurons in each layer.
    Networks can be saved to and then loaded from a text file.
    """
    def __init__(self, input_dimension, neurons_per_layer):
        self.inputData = None
        self.input_dimension = input_dimension
        self.neurons_per_layer = neurons_per_layer
        self.layers = [[] for i in range(len(neurons_per_layer))]
        for i in range(len(neurons_per_layer)):
            for j in range(neurons_per_layer[i]):
                if i == 0:
                    self.layers[i].append(   Neuron( [],
                                                    [(random.random()*2 - 1) for r in range(input_dimension)],
                                                    (random.random()*2 - 1) )
                                                    )
                else:
                    self.layers[i].append(   Neuron( self.layers[i-1],
                                                    [(random.random()*2 - 1) for r in range(neurons_per_layer[i-1])],
                                                    (random.random()*2 - 1) )
                                                    )
    ### one can initialize a network from a file
    @classmethod
    def loadFromFile(cls,filename):
        file = open(filename,"r")
        ### get network structure
        line = file.readline()
        input_dimension = int(line)
        line = file.readline()
        neurons_per_layer = [int(s) for s in line.strip('\n][ ').split(',')]

        ### initialize a network with correct structure but random values, and then set the correct weights etc.
        network = cls(input_dimension,neurons_per_layer)
        for i in range(len(neurons_per_layer)):
            line = file.readline()
            for j in range(neurons_per_layer[i]):
                line = file.readline()
                tokens = line.split(',')
                weights = []
                for s in tokens[0].strip('][').split(' '):
                    if s != '':
                        weights.append(float(s))

                network.layers[i][j].weights = np.array(weights)
                network.layers[i][j].bias = float(tokens[1])
        return network

    ### saves the network to a text file
    def saveToFile(self, filename):
        n_per_layer = str(self.input_dimension) + '\n' + str(self.neurons_per_layer)
        neuron_data = ''
        for i in range(len(self.neurons_per_layer)):
            neuron_data += '\nLayer ' + str(i + 1) + ':'
            for j in range(self.neurons_per_layer[i]):
                neuron = self.layers[i][j]
                neuron_data += '\n' + str(neuron.weights) + ',' + str(neuron.bias)
        ### write file data
        file = open(filename + '.txt', "w")
        file.write(n_per_layer + neuron_data)


    def evaluateInput(self, input):
        for neuron in self.layers[0]:
            neuron.inputValues = np.array(input)
            neuron.getOutputValue()
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.getInputValues()
                neuron.getOutputValue()
        return [neuron.outputValue for neuron in self.layers[-1]]

    def cost(self, training_sample):
        network_output = np.array(self.evaluateInput(training_sample[0]))
        desired_output = np.array(training_sample[1])
        return np.linalg.norm(network_output - desired_output)

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    ### the derivative of the sigmoid function
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    ### stochastic gradient descent training
    def SGD(self, training_data, epochs, batch_size, eta,display):
        n = len(training_data)
        for epoch in range(epochs):
            if (epoch % 10) == 0:
                print("  Epoch No: " + str(epoch))
            random.shuffle(training_data) # shuffle the data
            batches = [training_data[k:k+batch_size] for k in range(0,n,batch_size)] # split training data into batches
            for batch in batches:
                b_gradients = [np.zeros((len(l))) for l in self.layers]
                w_gradients = [np.zeros((len(l),len(l[0].weights))) for l in self.layers]
                for x,y in batch:
                    delta_b_gradients, delta_w_gradients = self.backprop(x,y)
                    b_gradients = [bg+dbg for bg, dbg in zip(b_gradients, delta_b_gradients)] # essentially: b_gradients += delta_b_gradients
                    w_gradients = [wg+dwg for wg, dwg in zip(w_gradients, delta_w_gradients)]

                # now update weights and biases once after processing a batch
                for i in range(len(self.layers)):
                    for j in range(len(self.layers[i])):
                        self.layers[i][j].bias -= (eta/len(batch))*b_gradients[i][j]
                        self.layers[i][j].weights -= (eta/len(batch))*w_gradients[i][j]
                if display != None:
                    display.drawWeights(self)

    def backprop(self,x,y):
        b_gradients = [np.zeros((len(l))) for l in self.layers]
        w_gradients = [np.zeros((len(l),len(l[0].weights))) for l in self.layers]
        ### first evaluate the training sample x ------------------
        activations = [x] # we append all activations as the input data is fed forward. Last item in list is the final output value
        curr_activation = x
        weighted_inputs = []

        for layer in self.layers:
            weights = np.array([neuron.weights for neuron in layer])
            biases  = np.array([neuron.bias  for neuron in layer])

            weighted_input = np.dot(weights, curr_activation) + biases
            weighted_inputs.append(weighted_input)
            curr_activation = self.sigmoid(weighted_input)
            activations.append(curr_activation)

        ### now pass backwards ----------------
        cost_derivative = activations[-1] - y # difference between predicted result and desired output 'y'
        error = cost_derivative * self.sigmoid_prime(weighted_inputs[-1])
        b_gradients[-1] = error

        for i in range(len(error)):
            for j in range(len(activations[-2])):
                w_gradients[-1][i][j] = error[i] * activations[-2][j]

        # loop back through the remaining layers
        for l in range(2, len(self.layers)):
            weights = np.array([neuron.weights for neuron in self.layers[-l+1]])
            weighted_input = weighted_inputs[-l]

            error = np.dot(weights.transpose(),error) * self.sigmoid_prime(weighted_input)
            b_gradients[-l] = error
            for i in range(len(error)):
                for j in range(len(activations[-2])):
                    w_gradients[-l][i][j] = error[i] * activations[-l-1][j]
        return b_gradients, w_gradients
