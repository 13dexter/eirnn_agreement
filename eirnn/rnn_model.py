import json
import multiprocessing
import os
import os.path as op
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np

import filenames
from utils import deps_from_tsv


class RNNModel(object):

    serialized_attributes = ['vocab_to_ints', 'ints_to_vocab', 'filename',
                             'X_train', 'Y_train', 'deps_train',
                             'X_test', 'Y_test', 'deps_test']

    def __init__(self, filename=None, serialization_dir=None,
                 batch_size=16, embedding_size=50, 
                 maxlen=50, prop_train=0.9, rnn_output_size=50,
                 mode='infreq_pos', vocab_file=filenames.vocab_file,
                 equalize_classes=False, criterion=None,
                 verbose=1):
        '''
        filename: TSV file with positive examples, or None if unserializing
        criterion: dependencies that don't meet this criterion are excluded
            (set to None to keep all dependencies)
        verbose: passed to Keras (0 = no, 1 = progress bar, 2 = line per epoch)
        '''
        self.filename = filename
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.prop_train = prop_train
        self.mode = mode
        self.rnn_output_size = rnn_output_size
        self.maxlen = maxlen
        self.equalize_classes = equalize_classes
        self.criterion = (lambda x: True) if criterion is None else criterion
        self.verbose = verbose
        # self.set_serialization_dir(serialization_dir)

    def log(self, message):
        print (message)

    def pipeline(self, train = True, load = False, model = ''):
        examples = self.load_examples()
        self.create_train_and_test(examples)
        self.create_model()
        if (load) :
            self.load_model(model)
        if (train) :
            self.train()
        self.results()

    def load_examples(self, n_examples=None):
        '''
        Set n_examples to some positive integer to only load (up to) that 
        number of examples
        '''
        self.log('Loading examples')
        if self.filename is None:
            raise ValueError('Filename argument to constructor can\'t be None')

        self.vocab_to_ints = {}
        self.ints_to_vocab = {}
        examples = []
        n = 0

        deps = deps_from_tsv(self.filename, limit=n_examples)

        for dep in deps:
            tokens = dep['sentence'].split()
            if len(tokens) > self.maxlen or not self.criterion(dep):
                continue

            tokens = self.process_single_dependency(dep)
            ints = []
            for token in tokens:
                if token not in self.vocab_to_ints:
                    # zero is for pad
                    x = self.vocab_to_ints[token] = len(self.vocab_to_ints) + 1
                    self.ints_to_vocab[x] = token
                ints.append(self.vocab_to_ints[token])

            examples.append((self.class_to_code[dep['label']], ints, dep))
            n += 1
            if n_examples is not None and n >= n_examples:
                break

        return examples

    def load_model(self, model) :
        self.model = torch.load(model)

    def train(self, n_epochs=10):
        self.log('Training')
        if not hasattr(self, 'model'):
            self.create_model()
        
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        prev_param = list(self.model.parameters())[0].clone()

        print(len(self.X_train))
        for epoch in range(100) :
            print('epoch : ', epoch)
            for index in range(len(self.X_train)) :
                if ((index+1) % 5000 == 0) :
                    print (index+1)
                    if (epoch == 0 or (index+1) % 10000 == 0):
                        self.results()
                        model_name = 'eirnn_model' + str(epoch) + '.pkl'
                        torch.save(self.model, model_name)
                
                self.model.zero_grad()
                output = self.model(self.X_train[index])
                if (self.Y_train[index] == 0) :
                    actual = torch.autograd.Variable(torch.tensor([0]), requires_grad=False)
                else :
                    actual = torch.autograd.Variable(torch.tensor([1]), requires_grad=False)
                
                loss = loss_function(output, actual)
                loss.backward(retain_graph=True)
                optimizer.step()

            param = list(self.model.parameters())[0].clone()
            print(prev_param)
            print(param)
            print(torch.equal(prev_param, param))
            prev_param = param.clone()

            self.results()
            model_name = 'eirnn_model' + str(epoch) + '.pkl'
            torch.save(self.model, model_name)