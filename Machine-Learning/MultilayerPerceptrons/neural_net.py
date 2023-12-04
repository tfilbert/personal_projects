"""
Implementation of a multi-layer perceptron
Author: Tyler Filbert
"""

import numpy as np
import math
import random
import data_processor as dp
import os


def sigmoid(x):
    return 1/(1+math.pow(math.e, -x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


class Perceptron_Weights:
    def __init__(self):
        # architecture is 784x256x1

        # 784x256
        self.weights_h = []
        # 256x1
        self.weights_o = []

        self.image_size = 784

        # Assign random edge weights
        for i in range(self.image_size):
            self.weights_h.append(list(np.random.uniform(-1, 1, 256)))
        for j in range(256):
            self.weights_o.append(random.uniform(-1, 1))

        # initialize bias terms
        self.bias_h = [-1] * 256
        self.bias_o = -1
        
        self.learning_rate = 0.3


    def train_net(self, input):
        """
        Given one instance of a data point, forward pass and back prop
        edge weights in nueral net
        """
        im_vals = input['image_vals']
        
        # ----- start forward pass ------
        # calculate hidden layer input
        h_in = np.dot(list(np.array(self.weights_h).transpose()), im_vals)
        h_in = np.add(self.bias_h, h_in)

        # calculate hidden layer output
        h_out = list(map(sigmoid, h_in))

        # generate full output
        o_out = sigmoid(np.dot(self.weights_o, h_out) + self.bias_o) 


        # ------- start back prop -------
        # calculate error
        err = float(input['class_label']) - o_out

        # calculate deltas
        delta_o = err * sigmoid_derivative(o_out)
        delta_h = np.dot(delta_o, self.weights_o) * list(map(sigmoid_derivative, h_in))

        # back prop edge weights from deltas
        self.weights_o += self.learning_rate * np.multiply(h_out, delta_o)
        self.weights_h += self.learning_rate * np.outer(im_vals, delta_h)

        # update bias terms
        self.bias_o += self.learning_rate * delta_o
        self.bias_h += np.dot(self.learning_rate, delta_h)

        return o_out

    
    def get_net_output(self, input):
        """
        Given an input, return the nets predicted output
        """
        im_vals = input['image_vals']
        
        # ----- start forward pass ------
        # calculate hidden layer input
        h_in = np.dot(list(np.array(self.weights_h).transpose()), im_vals)
        h_in = np.add(self.bias_h, h_in)

        # calculate hidden layer output
        h_out = list(map(sigmoid, h_in))

        # generate full output
        o_out = sigmoid(np.dot(self.weights_o, h_out) + self.bias_o)

        return o_out
    

    def test_net(self, test_data):
        """
        Iterate through all test data, getting net's predicted values
        and record accuracy
        """
        total = 0
        correct = 0

        # iterate through all images in test_data
        # use the nueral net to predict the ouput and compare with actual value
        for im_val in test_data:
            guessed = 0 if self.get_net_output(test_data[im_val]) < 0.5 else 1
            if guessed == float(test_data[im_val]['class_label']):
                correct += 1
            total += 1
        
        # report accuracy
        print('Total images tested: {tot}'.format(tot=total))
        print('Total correct: {tot}'.format(tot=correct))
        print('Total accuracy: {:.2%}'.format(correct/total))


def main():
    # process the training data
    train = dp.DataProcessor()
    train.get_example_values(os.path.join(os.path.dirname(__file__), 'data\mnist_train_0_1.csv'))

    weights = Perceptron_Weights()

    # train the model
    for im_num in train.all_sets:
        cur_im = train.all_sets[im_num]
        weights.train_net(cur_im)
    print('Completed learning from training data!\n')

    # process the test data
    test = dp.DataProcessor()
    test.get_example_values(os.path.join(os.path.dirname(__file__), 'data\mnist_test_0_1.csv'))

    weights.test_net(test.all_sets)


if __name__ == '__main__':
    main()