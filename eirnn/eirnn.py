import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math

def rectify(x):
    relu = nn.ReLU()
    return relu(x)

class EIRnnCell(nn.Module):
    def __init__(self, hidden_size, embedding_dim, output_size, rectify_inputs = True, var_input = 0.001, var_rec = 0.001, dt = 0.1, tau=100):
        super(EIRnnCell, self).__init__()

        self.N = hidden_size
        self.N_in = embedding_dim
        self.N_out = output_size
        self.rectify_inputs = rectify_inputs
        self.var_in = var_input
        self.var_rec = var_rec
        self.dt = dt
        self.tau = tau
        self.alpha = dt/tau

        # self.w_in = Variable(torch.randn(hidden_size, embedding_dim))
        # self.w_rec = rectify(Variable(torch.randn(hidden_size, hidden_size))
        # self.w_out = rectify(Variable(torch.randn(hidden_size, output_size)))

        self.w_in = rectify(Variable(torch.randn(hidden_size, embedding_dim))) #####
        self.w_rec_plus = rectify(Variable(torch.randn(hidden_size, hidden_size)))
        self.w_out = rectify(Variable(torch.randn(hidden_size, output_size)))
        self.d_rec = Variable(torch.zeros(hidden_size, hidden_size), requires_grad=False) #####
        for i in range(hidden_size) :
            if (i < 0.8*hidden_size):
                self.d_rec[i][i] = 1.0
            else:
                self.d_rec[i][i] = -1.0
        

        self.reset_parameters()

    def reset_parameters(self):
        """ 
        Initialize parameters (weights) like mentioned in the paper.
        """

    def forward(self, input_, prev_state):
        """
        Args:
            input_: A (batch, embedding_dim) tensor containing input
                features.
            prev_state: Contains the initial cell state, which has
                the size (batch, hidden_size).
        Returns:
            state: Tensor containing the next cell state.
        """
        rectified_pstate = rectify(prev_state)
        w_rec = torch.mm(self.w_rec_plus, self.d_rec)
        hidden_w = torch.mm(rectified_pstate, w_rec)
        input_w = torch.mv(self.w_in, input_)
        state = (1 - self.alpha) * prev_state + self.alpha * (hidden_w + input_w) + math.sqrt(2 * self.alpha * self.var_rec) * 0.001# guassian(0,1)
        rectified_nstate = rectify(state)
        [u, v] = state.size()
        self.w_in = rectify(self.w_in)
        self.w_rec_plus = rectify(self.w_rec_plus)
        self.w_out = rectify(self.w_out)
        output = torch.mm(rectified_nstate, self.w_out)
        return state, output

class EIRnn(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, vocab_size, num_layers = 1, batch_first = False, dropout=0):
        super(EIRnn, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_embedding_dim = embedding_dim if layer == 0 else hidden_size
            cell = EIRnnCell(embedding_dim = layer_embedding_dim, hidden_size = hidden_size, output_size = output_size)
            setattr(self, 'cell_{}'.format(layer), cell)
        
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.mlp = nn.Linear(output_size * num_layers, 2)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def forward(self, input_, state = None, batch_size = 1) :
        if state is None:
            state = torch.zeros([batch_size, self.hidden_size], dtype=torch.float)
        # state_n = []
        layer_output = None
        all_layers_last_out = []
        inputx = torch.from_numpy(input_)
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

            max_time = len(input_)
            output = []
            state_ = state
            for time in range(max_time):
                # print('t', time, inputx[time])
                next_state, out = cell(input_ = self.embedding_layer(Variable(inputx[time], requires_grad=False)), prev_state = state_)
                output.append(out)
                state_ = next_state
                last_out = out
                
            # layer_state = state
            layer_output = torch.stack(output, 0)
            all_layers_last_out.append(last_out)

            input_ = self.dropout_layer(layer_output)
            # state_n.append(layer_state)

        # output = layer_output
        # state_n = torch.stack(state_n, 0)
        output = torch.stack(all_layers_last_out, 0)
        output = output.view(max_time * self.num_layers)
        out2sig = self.mlp(output)
        # sigmoid_out = self.sigmoid(out2sig)
        softmax_out = self.softmax(out2sig)
        softmax_out = torch.stack([softmax_out], 0)
        return softmax_out
        