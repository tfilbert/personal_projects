"""
  A Convolutional Neural Network to Identify Corgi Breeds
    *woof*
  Author: Tyler Filbert
  Machine Learning Spring 2023 - University of Kentucky Programming Assignment 4
"""

from torch import flatten
from torch import from_numpy
from torch.nn import LogSoftmax
from torch.nn import MaxPool2d
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Linear
from torch.optim import Adam
from torch import nn
from torch import tensor
from torch import argmax
from torch import no_grad
import dataset as ds
import os


class CorgiNet(Module):
  def __init__(self, numChannels, classes):
    super(CorgiNet, self).__init__()

    # prev values out1=20, out2=50, out_feat = 500
    # initialize first set of Conv -> ReLU -> Pool Layers
    self.conv1 = Conv2d(in_channels=numChannels, out_channels=10, kernel_size=(5,5))
    self.relu1 = ReLU()
    self.maxpool1 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

    # initialize second set of Conv -> ReLU -> Pool Layers
    self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=(5,5))
    self.relu2 = ReLU()
    self.maxpool2 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

    # initialize only set of fully connected -> ReLU Layer
    self.fc1 = Linear(in_features=297680, out_features=250)
    self.relu3 = ReLU()

    # initialize softmax classifer
    self.fc2 = Linear(in_features=250, out_features=classes)
    self.logSoftmax = LogSoftmax(dim=1)

  def forward(self, x):
    # pass input through first set of Conv -> ReLU -> Pool Layers
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)

    # pass input through second set of Conv -> ReLU -> Pool Layers
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.maxpool2(x)

    # flatten the output from the previous layer and pass it through FC -> ReLU layer
    x = flatten(x, 1)
    x = self.fc1(x)
    x = self.relu3(x)

    # pass the output to softmax classifer for output predictions
    x = self.fc2(x)
    output = self.logSoftmax(x)

    # return the output predictions
    return output


def test_model(model, im_size, classes, class_str_to_int):
  # Load in the test data
  test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\testing_data')
  data_sets = ds.read_train_sets(test_path, im_size, classes, 0)
  test_set = data_sets.train

  num_examples = test_set._num_examples
  num_correct = 0

  # set model to evaluation mode
  model.eval()


  for i in range(test_set._num_examples):
    ex_image, ex_onehot, ex_filename, ex_label = test_set.next_batch(1)

    # convert class labels to integers and then to a tensor
    ex_label_ints = [class_str_to_int[c] for c in ex_label]
    ex_label_tensor = tensor(ex_label_ints)

    # convert batch_images to floats and then to a PyTorch tensor
    ex_image_tensor = from_numpy(ex_image).float()
    ex_image_tensor = ex_image_tensor.transpose(1, 3).transpose(2, 3)


    # do a forward pass and calculate the training loss
    pred = model(ex_image_tensor)

    # calculate the number of correct predictions
    # calculate number of correct predictions
    max_indices = argmax(pred, dim=1)
    match = (max_indices == ex_label_tensor)
    num_correct += match.sum().item()

  print('Test Accuracy: ', num_correct/num_examples)
    

def main():
  IMAGE_SIZE = 500
  classes = ['pembroke', 'cardigan']
  class_int_to_str = {0: 'pembroke', 1: 'cardigan'}
  class_str_to_int = {'pembroke': 0, 'cardigan': 1}
  alpha = 0.0002
  EPOCHS = 3
  BATCH_SIZE = 5
  VALIDATION_SIZE = 40

  train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\training_data')

  data_sets = ds.read_train_sets(train_path, IMAGE_SIZE, classes, VALIDATION_SIZE)

  training_set = data_sets.train
  validation_set = data_sets.valid

  model = CorgiNet(numChannels=3, classes=len(classes))
  opt = Adam(model.parameters(), lr=alpha)
  lossFn = nn.CrossEntropyLoss()
  
  for e in range(EPOCHS):
    # put model in training mode
    model.train()

    # initialize the running total for training and validation set loss
    total_train_loss = 0
    total_vali_loss = 0

    # initialize the number of correct predictions made in training and validation
    train_correct = 0
    vali_correct = 0

    # initalize how many batches must be made
    num_batches = training_set._num_examples // BATCH_SIZE

    # initalize how many examples are in train and validation sets
    train_total = num_batches * BATCH_SIZE

    for i in range(num_batches):
      # get the next batch's images, labels, image names, and class labels
      batch_images, batch_onehots, batch_filenames, batch_labels = training_set.next_batch(BATCH_SIZE)

      # convert class labels to integers and then to a tensor
      batch_label_ints = [class_str_to_int[c] for c in batch_labels]
      batch_labels_tensor = tensor(batch_label_ints)

      # convert batch_images to floats and then to a PyTorch tensor
      batch_images_tensor = from_numpy(batch_images).float()
      batch_images_tensor = batch_images_tensor.transpose(1, 3).transpose(2, 3)

      # do a forward pass and calculate the training loss
      pred = model(batch_images_tensor)
      loss = lossFn(pred, batch_labels_tensor)

      # zero gradients, backpropagate, and update the weights
      opt.zero_grad()
      loss.backward()
      opt.step()

      # add the loss to the total training loss
      total_train_loss += loss

      # calculate number of correct predictions
      max_indices = argmax(pred, dim=1)
      match = (max_indices == batch_labels_tensor)
      train_correct += match.sum().item()

      #print('loss: ', loss.item())
      #print('predictions: \n', max_indices)
      #print('labels: \n', batch_labels_tensor)

    print('Epoch: ', e+1)
    print('Training Accuray: ', train_correct/train_total)
    with no_grad():
      # set the model to evaluation mode
      model.eval()

      # loop through validation set
      for i in range(VALIDATION_SIZE):
        ex_image, ex_onehot, ex_filename, ex_label = validation_set.next_batch(1)

        # convert class labels to integers and then to a tensor
        ex_label_ints = [class_str_to_int[c] for c in ex_label]
        ex_label_tensor = tensor(ex_label_ints)

        # convert batch_images to floats and then to a PyTorch tensor
        ex_image_tensor = from_numpy(ex_image).float()
        ex_image_tensor = ex_image_tensor.transpose(1, 3).transpose(2, 3)


        # do a forward pass and calculate the training loss
        pred = model(ex_image_tensor)
        loss = lossFn(pred, ex_label_tensor)

        # calculate the number of correct predictions
        # calculate number of correct predictions
        max_indices = argmax(pred, dim=1)
        match = (max_indices == ex_label_tensor)
        vali_correct += match.sum().item()
    
      print('Validation Accuracy: ', vali_correct/VALIDATION_SIZE)

    
  test_model(model, IMAGE_SIZE, classes, class_str_to_int)



if __name__ == '__main__':
  main()