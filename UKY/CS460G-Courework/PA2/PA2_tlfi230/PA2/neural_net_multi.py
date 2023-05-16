"""
Implementation of a multi-layer perceptron for classifying images
for 0-4
Author: Tyler Filbert
"""

import numpy as np
import math
import data_processor as dp
import os


def sigmoid(x):
    return 1/(1+math.pow(math.e, -x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


class Perceptron_Weights:
    def __init__(self):
        # architecture is 784x256x5

        # 784x256
        self.weights_h = []

        # 256x5
        self.weights_o = []

        self.image_size = 784

        # Assign random edge weights
        for i in range(self.image_size):
            self.weights_h.append(list(np.random.uniform(-1, 1, 256)))
        for j in range(256):
            self.weights_o.append(list(np.random.uniform(-1, 1, 5)))

        # initialize bias terms
        self.bias_h = [-1] * 256
        self.bias_o = [-1] * 5
        
        self.learning_rate = 0.40


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

        o_in = np.dot(list(np.array(self.weights_o).transpose()), h_out)
        o_in = np.add(self.bias_o, o_in)
        # generate full output
        o_out = list(map(sigmoid, o_in))
        # ------ completed forward pass -------


        # ------- start back prop -------
        # calculate error
        correct_vector = [0, 0, 0, 0, 0]
        correct_vector[int(input['class_label'])] = 1
        err = np.subtract(correct_vector, o_out)

        # calculate deltas
        delta_o = err * list(map(sigmoid_derivative, o_in))
        delta_h = np.dot(self.weights_o, delta_o) * list(map(sigmoid_derivative, h_in))

        # back prop edge weights from deltas
        self.weights_o += self.learning_rate * np.multiply(o_out, delta_o)
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

        o_in = np.dot(list(np.array(self.weights_o).transpose()), h_out)
        o_in = np.add(self.bias_o, o_in)

        # generate full output
        o_out = list(map(sigmoid, o_in))

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
            guessed = self.get_net_output(test_data[im_val])
            max_index = np.argmax(guessed)
            if max_index == float(test_data[im_val]['class_label']):
                correct += 1
            total += 1
        
        # report accuracy
        print('Total images tested: {tot}'.format(tot=total))
        print('Total correct: {tot}'.format(tot=correct))
        print('Total accuracy: {:.2%}'.format(correct/total))


def main():
    # process the training data
    train = dp.DataProcessor()
    train.get_example_values(os.path.join(os.path.dirname(__file__), 'data\mnist_train_0_4.csv'))

    weights = Perceptron_Weights()

    # train the model
    for im_num in train.all_sets:
        cur_im = train.all_sets[im_num]
        weights.train_net(cur_im)
    print('Completed learning from training data!\n')

    # process the test data
    test = dp.DataProcessor()
    test.get_example_values(os.path.join(os.path.dirname(__file__), 'data\mnist_test_0_4.csv'))

    weights.test_net(test.all_sets)


if __name__ == '__main__':
    main()