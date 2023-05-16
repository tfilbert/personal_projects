# Recurrent Neural Net Project 3 CS460G Spring 23
# Author: Tyler Filbert

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import data_handler as dh
import os

# batch here
def set_sequences(sentences):
    input_sequence = []
    target_sequence = []
    for i in range(len(sentences)):
        # Remove the last character from the input sequence
        input_sequence.append(sentences[i][:-1])
        # Remove the first element from target sequences
        target_sequence.append(sentences[i][1:])

    return input_sequence, target_sequence


def convert_chars_to_ints(sentences, charInt, input_sequence, target_sequence):
    for i in range(len(sentences)):
        input_sequence[i] = [charInt[character] for character in input_sequence[i]]
        target_sequence[i] = [charInt[character] for character in target_sequence[i]]

    return input_sequence, target_sequence


def create_one_hot(sequence, vocab_size):
    encoding = np.zeros((1, len(sequence), vocab_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0, i, sequence[i]] = 1

    return encoding


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #Define the network!
        #Batch first defines where the batch parameter is in the tensor
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        hidden_state = self.init_hidden(x.size(0))
        output, hidden_state = self.rnn(x, hidden_state)
        output = self.fc(output)

        return output, hidden_state
    
    def init_hidden(self, batch_size):
        #Also a note, pytorch, by default, wants the batch index
        # to be the middle dimension here.
        #So it looks like (row, BATCH, column)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden


def predict(model, character, charInt, intChar, vocab_size):
    # predict the next character
    characterInput = np.array([charInt[c] for c in character])
    characterInput = create_one_hot(characterInput, vocab_size)
    characterInput = torch.from_numpy(characterInput)
    out, hidden = model(characterInput)
    out = out.squeeze(dim=0)

    prob = nn.functional.softmax(out[-1], dim=0).data

    character_index = torch.max(prob, dim=0)[1].item()

    return intChar[character_index], hidden


def sample(model, out_len, start, charInt, intChar, vocab_size):
    characters = [ch for ch in start]
    current_size = out_len - len(characters)
    for i in range(current_size):
        character, hidden_state = predict(model, characters, charInt, intChar, vocab_size)
        characters.append(character)
    
    return ''.join(characters)


def main():
    # Read in data
    data_preparer = dh.DataHandler()
    data_preparer.process_txt(os.path.join(os.path.dirname(__file__), 'tiny-shakespeare.txt'))
    data_preparer.get_chars(data_preparer.sentences)
    data_preparer.set_vocab(data_preparer.chars)
    data_preparer.set_vocab_size(data_preparer.charInt)

    # as number of sentences
    BATCH_SIZE = 5
    EPOCHS = 5
    
    # Make num of sentences divisible by batch_count, remove any stragglers
    max_sentence_len = max([len(sen) for sen in data_preparer.sentences])
    fill_count = BATCH_SIZE - (len(data_preparer.sentences) % BATCH_SIZE)
    del data_preparer.sentences[-fill_count:]

    # Define model
    model = RNNModel(data_preparer.vocab_size, data_preparer.vocab_size, 100, 1)

    #Define loss
    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())
    training_loader = torch.utils.data.DataLoader(data_preparer.sentences, batch_size=BATCH_SIZE, shuffle=True)
        
    for epoch in range(EPOCHS):
        loss_per_epoch = 0
        num_examples_per_epoch = 0

        # per epoch training
        for i, data in enumerate(training_loader):
            input_sequence, target_sequence = set_sequences(data)
            input_sequence, target_sequence = convert_chars_to_ints(data,
                                                            data_preparer.charInt,
                                                            input_sequence,
                                                            target_sequence)
            optimizer.zero_grad()

            max_sentence_len = max([len(sen) for sen in input_sequence])

            inputs = []
            labels = []
            # Process the inputs and targets
            for j in range(len(input_sequence)):
                # Create one hots for inputs and save with padding
                x = torch.from_numpy(create_one_hot(input_sequence[j], data_preparer.vocab_size))
                x = torch.nn.functional.pad(x, (0, 0, 0, max_sentence_len-x.shape[1], 0, 0), mode='constant', value=0)
                inputs.append(x)

                # save the target, with padding
                y = torch.Tensor(target_sequence[j])
                pad = max(max_sentence_len - len(y), 0)
                y = torch.nn.functional.pad(y, (0, pad), mode='constant', value=data_preparer.charInt['-PAD-'])
                labels.append(y.view(-1).long())
            
            # Process model output and reshape
            output, hidden = model(torch.cat(inputs, dim=0))
            labels = torch.stack(labels)
            output = output.flatten(end_dim=1)
            labels = labels.flatten()

            # calcualte loss
            loss_value = loss(output, labels)
            
            # calculate gradient
            loss_value.backward()

            # update weights
            optimizer.step()

            loss_per_epoch += loss_value.item()
            num_examples_per_epoch += 1

        print('EPOCH NUM:', epoch + 1, ' --------> AVG LOSS: ', '{:.4f}'.format(loss_per_epoch/num_examples_per_epoch))

    print('Completed Training!')

    print(sample(model, 50, 'QUEEN:', data_preparer.charInt, data_preparer.intChar, data_preparer.vocab_size))

if __name__ == '__main__':
    main()


