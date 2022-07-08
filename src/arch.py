import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from src.postprocess import summaryInfo
from src.param_act_func import param_sigmoid, param_relu


class Autoencoder(nn.Module):
    def __init__(self, resolution, args):
        super(Autoencoder, self).__init__()
        self.layers = args.layers
        self.act_code = args.act_code
        self.act_hid = args.act_hid
        self.act_out = args.act_out
        self.alpha_relu = args.alpha_relu
        self.alpha_sigmoid = args.alpha_sigmoid
        self.initialisation = args.initialisation
        self.dropout = args.dropout
        self.dropout_prob = args.dropout_prob
        self.verbose = args.verbose
        self.param_relu = param_relu(self.alpha_relu)
        self.param_sigmoid = param_sigmoid(self.alpha_sigmoid)

        # Add input layer with image resolution as dimensions
        self.layers = [resolution[0]*resolution[1]] + self.layers

        # Encoder
        steps = len(self.layers)-1
        self.encoder = nn.ModuleList()
        self.activation_encoder = []
        self.encoder.append(nn.Flatten())
        self.activation_encoder.append('linear')
        if self.dropout:
            self.encoder.append(nn.Dropout(p=self.dropout_prob))
            self.activation_encoder.append('linear')
        # Encoder hidden layers
        for k in range(steps-1):
            self.encoder.append(nn.Linear(self.layers[k], self.layers[k+1]))
            self.activation_encoder.append(self.act_hid)
            if self.dropout:
                self.encoder.append(nn.Dropout(p=self.dropout_prob))
                self.activation_encoder.append('linear')
        # Code
        self.encoder.append(nn.Linear(self.layers[-2], self.layers[-1]))
        self.activation_encoder.append(self.act_code)

        # Decoder
        self.decoder = nn.ModuleList()
        self.activation_decoder = []
        for k in range(steps-1):
            self.decoder.append(nn.Linear(self.layers[steps-k], self.layers[steps-k-1]))
            self.activation_decoder.append(self.act_hid)
            if self.dropout:
                self.decoder.append(nn.Dropout(p=self.dropout_prob))
                self.activation_decoder.append('linear')
        # Add last decoder layer
        self.decoder.append(nn.Linear(self.layers[1], self.layers[0]))
        self.activation_decoder.append(self.act_out)

        # Weight initialisation  
        self.__weight_init(self.encoder)
        self.__weight_init(self.decoder)
        
        self.__summary()

    def __summary(self):
        name = 'results/archTable.png'
        data = [
            ['all layers', self.layers],
            ['weight init', self.initialisation],
            ['act funct hid layers', self.act_hid],
            ['act funct code layer', self.act_code],
            ['act funct last layer', self.act_out],
        ]
        if self.act_hid == 'param_relu' or self.act_code == 'param_relu':
            data.append(['relu initial alpha', self.alpha_relu])
        if self.act_out == 'param_sigmoid':
            data.append(['sigmoid initial alpha', self.alpha_sigmoid])
        data.append(['dropout', self.dropout])
        if self.dropout:
            data.append(['dropout prob', self.dropout_prob])
        
        summaryInfo(data, name, self.verbose)

    def __activation_function(self, x, activation):
        if activation == 'linear': x = x
        elif activation == 'sigmoid': x = torch.sigmoid(x)
        elif activation == 'relu': x = F.relu(x)
        elif activation == 'rrelu': x = F.rrelu(x)
        elif activation == 'tanh': x = torch.tanh(x)
        elif activation == 'elu': x = F.elu(x)
        elif activation == 'param_sigmoid': x = self.param_sigmoid(x)
        elif activation == 'param_relu': x = self.param_relu(x)
        else: raise NotImplementedError
        return x
    
    def __weight_init(self, struct):
        for layer in struct:
            if layer._get_name() == 'Linear':
                if self.initialisation == 'zeros': 
                    init.constant_(layer.bias, 0)
                    init.constant_(layer.weight, 0)
                elif self.initialisation == 'xavier_normal': 
                    init.constant_(layer.bias, 0)
                    init.xavier_normal_(layer.weight)
                elif self.initialisation == 'xavier_uniform': 
                    init.constant_(layer.bias, 0)
                    init.xavier_uniform_(layer.weight)
                elif self.initialisation == 'kaiming_normal': 
                    init.constant_(layer.bias, 0)
                    init.kaiming_normal_(layer.weight)
                elif self.initialisation == 'kaiming_uniform': 
                    init.constant_(layer.bias, 0)
                    init.kaiming_uniform_(layer.weight)
                elif self.initialisation == 'sparse': 
                    init.constant_(layer.bias, 0)
                    init.sparse_(layer.weight, sparsity = 0.5)
                else:
                    raise NotImplementedError
    
    def encode(self, x):
        i = 0
        for layer in self.encoder:
            x = layer(x)
            x = self.__activation_function(x, self.activation_encoder[i])
            i += 1
        return x

    def decode(self, x):
        i = 0
        for layer in self.decoder:
            x = layer(x)
            x = self.__activation_function(x, self.activation_decoder[i])
            i += 1
        return x

    def forward(self, x):
        code = self.encode(x)
        x_nn = self.decode(code)
        return x_nn, code