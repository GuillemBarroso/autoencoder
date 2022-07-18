import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from src.param_act_func import param_sigmoid, param_relu

class Autoencoder(nn.Module):
    def __init__(self, data, args):
        super(Autoencoder, self).__init__()
        self.resolution = data.resolution
        self.resolution_mu = data.resolution_mu
        self.n_mus = data.n_mus
        self.layers = args.layers
        self.layers_mu = args.layers_mu
        self.dropout = args.dropout
        self.dropout_prob = args.dropout_prob
        self.act_hid = args.act_hid
        self.act_code = args.act_code
        self.act_out = args.act_out
        self.initialisation = args.initialisation
        self.param_activation = False
        self.mode = args.mode

        # Add input layer with image resolution as dimensions
        self.layers = [self.resolution[0]*self.resolution[1]] + self.layers
        self.layers_mu = [self.n_mus] + self.layers_mu
        self.steps = len(self.layers)-1

        if self.mode == 'standard':
            self.loss_names = ['total loss', 'image loss']
        elif self.mode == 'parametric':
            self.loss_names = ['total loss', 'image nn loss', 'image mu loss', 'code loss']
        else:
            raise NotImplementedError

        if args.reg:
            self.loss_names.append('reg loss')

        if 'param_relu' in [self.act_out, self.act_hid, self.act_code]:
            self.param_relu = param_relu(self.alpha_relu)
            self.param_activation = True
        if 'param_sigmoid' in [self.act_out, self.act_hid, self.act_code]:
            self.param_sigmoid = param_sigmoid(self.alpha_sigmoid)
            self.param_activation = True

    def layersLoop(self, x, layers, activations):
            for i, layer in enumerate(layers):
                x = self.__activation_function(layer(x), activations[i])
            return x

    def weight_init(self, struct):
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

class Encoder(Autoencoder):
    def __init__(self, data, args):
        super(Encoder, self).__init__(data, args)
        self.build()
        summary(self, self.resolution)

    def build(self):
        # Encoder
        self.encoder = nn.ModuleList()
        self.activation_encoder = []
        self.encoder.append(nn.Flatten())
        self.activation_encoder.append('linear')
        if self.dropout:
            self.encoder.append(nn.Dropout(p=self.dropout_prob))
            self.activation_encoder.append('linear')
        # Encoder hidden layers
        for k in range(self.steps-1):
            self.encoder.append(nn.Linear(self.layers[k], self.layers[k+1]))
            self.activation_encoder.append(self.act_hid)
            if self.dropout:
                self.encoder.append(nn.Dropout(p=self.dropout_prob))
                self.activation_encoder.append('linear')
        # Code
        self.encoder.append(nn.Linear(self.layers[-2], self.layers[-1]))
        self.activation_encoder.append(self.act_code)

        # Weight initialisation  
        self.weight_init(self.encoder)

    def encode(self, x):
        return self.layersLoop(x, self.encoder, self.activation_encoder)

    def forward(self, x):
        return self.encode(x)

class Decoder(Autoencoder):
    def __init__(self, data, args):
        super(Decoder, self).__init__(data, args)
        self.build()
        summary(self, (1,torch.tensor(self.layers[-1])))

    def build(self):
        # Decoder
        self.decoder = nn.ModuleList()
        self.activation_decoder = []
        self.decoder.append(nn.Flatten())
        self.activation_decoder.append('linear')
        for k in range(self.steps-1):
            self.decoder.append(nn.Linear(self.layers[self.steps-k], self.layers[self.steps-k-1]))
            self.activation_decoder.append(self.act_hid)
            if self.dropout:
                self.decoder.append(nn.Dropout(p=self.dropout_prob))
                self.activation_decoder.append('linear')
        # Add last decoder layer
        self.decoder.append(nn.Linear(self.layers[1], self.layers[0]))
        self.activation_decoder.append(self.act_out)

        # Weight initialisation  
        self.weight_init(self.decoder)

    def decode(self, x):
        return self.layersLoop(x, self.decoder, self.activation_decoder)

    def forward(self, x):
        return self.decode(x)


class Parameter(Autoencoder):
    def __init__(self, data, args):
        super(Parameter, self).__init__(data, args)
        self.build()
        summary(self, self.resolution_mu)

    def build(self):
        # Parameter
        steps_mu = len(self.layers_mu)-1
        self.parameter = nn.ModuleList()
        self.activation_param = []
        for k in range(steps_mu-1):
            self.parameter.append(nn.Linear(self.layers_mu[k], self.layers_mu[k+1]))
            self.activation_param.append(self.act_hid)
        self.parameter.append(nn.Linear(self.layers_mu[-2], self.layers_mu[-1]))
        self.activation_param.append(self.act_code)

        # Weight initialisation  
        self.weight_init(self.parameter)

    def param(self, x):
        return self.layersLoop(x, self.parameter, self.activation_param)

    def forward(self, x):
        return self.param(x)