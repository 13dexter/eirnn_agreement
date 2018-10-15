import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import six

from rnn_model import RNNModel
from eirnn import EIRnn
from utils import gen_inflect_from_vocab, dependency_fields

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

class RNNAcceptor(RNNModel):

    def create_train_and_test(self, examples):
        d = [[], []]
        for i, s, dep in examples:
            d[i].append((i, s, dep))
        random.seed(1)
        random.shuffle(d[0])
        random.shuffle(d[1])
        if self.equalize_classes:
            l = min(len(d[0]), len(d[1]))
            examples = d[0][:l] + d[1][:l]
        else:
            examples = d[0] + d[1]
        random.shuffle(examples)

        Y, X, deps = zip(*examples)
        Y = np.asarray(Y)
        print(X[0])
        X = pad_sequences(X, maxlen = self.maxlen)
        print(X[0])
        n_train = int(self.prop_train * len(X))
        self.X_train, self.Y_train = X[:n_train], Y[:n_train]
        self.X_test, self.Y_test = X[n_train:n_train+20000], Y[n_train:n_train+20000]
        self.deps_train = deps[:n_train]
        self.deps_test = deps[n_train:n_train+20000]

    def create_model(self):
        self.log('Creating model')
        print('vocab size', len(self.vocab_to_ints))
        self.model = EIRnn(embedding_dim = self.embedding_size, hidden_size=self.embedding_size, output_size=self.rnn_output_size, vocab_size=len(self.vocab_to_ints)+1)

    # def evaluate(self):
    #     return self.model.evaluate(self.X_test, self.Y_test,
    #                                batch_size=self.batch_size)

    def results(self):
        self.log('Processing test set')
        predicted = []
        print (len(self.X_train), len(self.X_test))
        with torch.no_grad():
            for index in range(len(self.X_test)) :
                pred = self.model(np.asarray(self.X_test[index], dtype=int))
                if (pred[0][0] > pred[0][1]) :
                    predicted.append([0])
                else :
                    predicted.append([1])
            # predicted, state = self.model(self.X_test)
            recs = []
            columns = ['correct', 'prediction', 'label'] + dependency_fields
            for dep, prediction in zip(self.deps_test, predicted):
                prediction = self.code_to_class[prediction[0]]
                recs.append((prediction == dep['label'], prediction, dep['label']) + tuple(dep[x] for x in dependency_fields))
        
        self.test_results = pd.DataFrame(recs, columns=columns)
        xxx = self.test_results['correct']
        print('Accuracy : ' + str(sum(xxx)))

class PredictVerbNumber(RNNAcceptor):

    def __init__(self, *args, **kwargs):
        RNNAcceptor.__init__(self, *args, **kwargs)
        self.class_to_code = {'VBZ': 0, 'VBP': 1}
        self.code_to_class = {x: y for y, x in self.class_to_code.items()}

    def process_single_dependency(self, dep):
        dep['label'] = dep['verb_pos']
        v = int(dep['verb_index']) - 1
        tokens = dep['sentence'].split()[:v]
        return tokens