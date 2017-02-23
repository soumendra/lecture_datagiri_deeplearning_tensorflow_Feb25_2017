"""
Source: https://www.amazon.in/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G
Make Your own Neural Network.

"""


import numpy as np
from scipy import special

class neuralNetwork:
    #initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate
        self.activation_function = lambda x:special.expit(x)

        pass


    def train(self, input_list, targets_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(targets_list, ndmin =2).T

        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        #calculate the signal emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #output error
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        #update the errors
        self.who += self.lr*np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors* hidden_outputs*(1.0-hidden_outputs)), np.transpose(inputs))

        pass

        #query the neural network
    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
