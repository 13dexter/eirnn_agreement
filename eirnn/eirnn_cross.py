import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math

def rectify(x):
    relu = nn.ReLU()
    return relu(x)
    # return x

class EIRnnModule(nn.Module):
    def __init__(self, input_units, output_units, hidden_units, embedding_dim = 50, rectify_inputs = True, var_input = 0.01**2, var_rec = 0.15**2, dt = 0.5, tau=100):
        super(EIRnnModule, self).__init__()

        self.n = hidden_units
        self.n_in = input_units
        self.n_out = output_units
        self.embedding_dim = embedding_dim
        self.rectify_inputs = rectify_inputs
        self.var_in = var_input
        self.var_rec = var_rec
        self.dt = dt
        self.tau = tau
        self.alpha = dt/tau

        self.w_in = rectify(Variable(torch.randn(hidden_units, input_units), requires_grad = True))
        self.w_rec = rectify(Variable(torch.randn(hidden_units, hidden_units), requires_grad = True))
        self.w_out = rectify(Variable(torch.randn(output_units, hidden_units), requires_grad = True))
        self.d_rec = Variable(torch.zeros(hidden_units, hidden_units), requires_grad=False)
        self.no_self_connect = Variable(torch.ones(hidden_units, hidden_units), requires_grad=False)

        for i in range(hidden_units) :
            self.no_self_connect = 0.0
            if (i < 0.8*hidden_units):
                self.d_rec[i][i] = 1.0
            else:
                self.d_rec[i][i] = -1.0
        

        self.reset_parameters()

    def reset_parameters(self):
        """ 
        Initialize parameters (weights) like mentioned in the paper.
        """

    def forward(self, input_, states):
        """
        Args:
            input_: A (embedding_dim, input_units) tensor containing input
                features.
            states: Contains the initial cell states, which has
                the size (embedding_dim, hidden_units).
        Returns:
            state: Tensor containing the next cell state.
        """
        # Rectify to upholds Dale's
        self.w_in = rectify(self.w_in)
        self.w_rec = rectify(self.w_rec)
        self.w_out = rectify(self.w_out)
        rectified_states = rectify(states)
        # rectified_states = states

        # No self connections 
        self.w_rec = self.w_rec * self.no_self_connect

        # Apply Dale's on recurrent weights
        w_rec_dale = torch.mm(self.w_rec, self.d_rec)
        # w_rec_dale = self.w_rec

        # print('W_in : ', self.w_in)
        # print('W_rec : ', self.w_rec)
        # print('W_out : ', self.w_out)
        # print('D_rec : ', self.d_rec)
        # print('Dale : ', w_rec_dale)

        hidden_update = torch.mm(w_rec_dale, rectified_states)
        input_update = torch.mm(self.w_in, input_)
        states = (1 - self.alpha) * states + self.alpha * (hidden_update + input_update) #+ math.sqrt(2 * self.alpha * self.var_rec) * 0.001 # guassian(0,1)
        rectified_states = rectify(states)
        # rectified_states = states
        # [u, v] = state.size()
        outputs = torch.mm(self.w_out, rectified_states)
        return states, outputs

class EIRnn(nn.Module):
    def __init__(self, embedding_dim, input_units, hidden_units, output_units, vocab_size, num_layers = 1, dropout=0):
        super(EIRnn, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_layers = num_layers
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_units = input_units if layer == 0 else hidden_units
            cell = EIRnnModule(input_units = input_units, output_units = output_units, hidden_units = hidden_units)
            setattr(self, 'cell_{}'.format(layer), cell)
        
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        # self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(output_units * embedding_dim * num_layers, 2)
        # self.softmax = nn.Softmax(dim=0)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def forward(self, input_, max_time = 15, input_once = True, states_init = None) :
        if states_init is None:
            states_init = torch.zeros([self.hidden_units, self.embedding_dim], dtype=torch.float)
        # state_n = []
        layer_output = None
        all_layers_last_output = []
        input0 = torch.zeros(len(input_), dtype=torch.long)
        inputx = torch.tensor(input_, requires_grad = False)
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

            output = []
            states = states_init
            for time in range(max_time):
                if (input_once and time != 0) :
                    next_states, outs = cell(input_ = self.embedding_layer(input0), states = states)
                else :
                    next_states, outs = cell(input_ = self.embedding_layer(inputx), states = states)

                output.append(outs)
                states = next_states
                last_outs = outs
            
            # layer_states = states
            layer_output = torch.stack(output, 0)
            all_layers_last_output.append(last_outs)

            # input_ = self.dropout_layer(layer_output)
            # state_n.append(layer_states)

        # state_n = torch.stack(state_n, 0)
        output = torch.stack(all_layers_last_output, 0)
        output = output.view(self.output_units * self.embedding_dim * self.num_layers)
        softmax_out = self.linear(output)
        # softmax_out = self.softmax(out2softmax_in)
        softmax_out = torch.stack([softmax_out], 0)
        return softmax_out
        