import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from src.postprocess import summaryInfo


class Autoencoder(nn.Module):
    def __init__(self, resolution, args):
        super(Autoencoder, self).__init__()
        self.layers = args.layers
        self.activation = args.activation
        self.initialisation = args.initialisation
        self.verbose = args.verbose
        self.activation_encoder = ['linear'] + (len(self.layers)-1)*[self.activation] + ['linear']
        self.activation_decoder = (len(self.layers)-1)*[self.activation] + ['sigmoid']

        # Add input layer with image resolution as dimensions
        self.layers = [resolution[0]*resolution[1]] + self.layers

        # Create ModuleList to store layers that will be used later to build the autoencoder
        steps = len(self.layers)-1
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Flatten())
        for k in range(steps):
            self.encoder.append(nn.Linear(self.layers[k], self.layers[k+1]))
        
        self.decoder = nn.ModuleList()
        for k in range(steps):
            self.decoder.append(nn.Linear(self.layers[steps-k], self.layers[steps-k-1]))

        # Weight initialisation  
        self.__weight_init(self.encoder)
        self.__weight_init(self.decoder)
        
        self.__summary()

    def __summary(self):
        name = 'results/archTable.png'
        data = [
            ['all layers', self.layers],
            ['act funct hid layers', self.activation],
            ['weight init', self.initialisation],
        ]
        summaryInfo(data, name, self.verbose)

    def __activation_function(self, x, activation):
        if activation == 'linear': x = x
        elif activation == 'sigmoid': x = torch.sigmoid(x)
        elif activation == 'relu': x = F.relu(x)
        elif activation == 'rrelu': x = F.rrelu(x)
        elif activation == 'tanh': x = torch.tanh(x)
        elif activation == 'elu': x = F.elu(x)
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