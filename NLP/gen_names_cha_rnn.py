from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import os
import torch.nn as nn
import torch
import random
import time

# Preparing the data
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # plus EOS marker


def find_files(path): return glob.glob(path)


def unicode2ascii(s):
    """
    Turn a Unicode string to plain ASCII.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_lines(filename):
    """
    Read a file and split into lines.
    """
    file_line = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode2ascii(line) for line in file_line]


# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for fn in find_files('../../PyTorch/data/data/names/*.txt'):
    category = os.path.splitext(os.path.basename(fn))[0]
    all_categories.append(category)
    lines = read_lines(fn)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
                       'from https://download.pytorch.org/tutorial/data.zip and extract it to '
                       'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicode2ascii("O'Néàl"))


# Creating the Network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + output_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + output_size, output_size)
        self.o2o = nn.Linear(input_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category_input, x, hidden):
        input_combined = torch.cat((category_input, x, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# Preparing for Training
def ran_choice(l):
    """
    Random item from a list.
    """
    return l[random.randint(0, len(l) - 1)]


def ran_input():
    """
    Get a random category and random line from that category
    """
    category_sample = ran_choice(all_categories)
    line = ran_choice(category_lines[category])
    return category_sample, line


def category2tensor(category_input):
    """
    One-hot vector for category
    """
    i = all_categories.index(category_input)
    tensor = torch.zeros(1, n_categories)
    tensor[0][i] = 1
    return tensor


def input2tensor(line_input):
    """
    One-hot matrix of first to last letters (not including EOS) for input.
    """
    tensor = torch.zeros(len(line_input), 1, n_letters)
    for i in range(len(line_input)):
        letter = line_input[i]
        tensor[i][0][all_letters.find(letter)] = 1
    return tensor

import numpy as np

def target2tensor(line_input):
    """
    LongTensor of second letter to end (EOS) for target.
    """
    letter_indexes = [all_letters.find(line_input[i]) for i in range(1, len(line_input))]
    letter_indexes.append(n_letters - 1)   # EOS
    return torch.LongTensor(1)


def ran_sample():
    """
    Make category, input, and target tensors from a random category, line pair
    """
    category_input, line_input = ran_input()
    category_tensor = category2tensor(category_input)
    input_tensor = input2tensor(line_input)
    target_tensor = target2tensor(line_input)
    return category_tensor, input_tensor, target_tensor


# Training the Network
