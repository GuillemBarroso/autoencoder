import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from src.param_act_func import param_sigmoid, param_relu
from src.postprocess import summaryInfo

class Autoencoder():
    def __init__(self, data, args):
        self.resolution = data.resolution
        self.resolution_mu = data.resolution_mu
        self.n_mus = data.n_mus
        self.name = data.name
        self.fig_path = data.fig_path
        self.layers = args.layers
        self.layers_mu = args.layers_mu
        self.dropout = args.dropout
        self.dropout_prob = args.dropout_prob
        self.act_hid = args.act_hid
        self.act_code = args.act_code
        self.act_out = args.act_out
        self.initialisation = args.initialisation
        self.mode = args.mode
        self.verbose = args.verbose
        self.save_fig = args.save_fig
        self.param_activation = False

        # Add input layer with image resolution as dimensions
        self.layers = [self.resolution[0]*self.resolution[1]] + self.layers
        self.layers_mu = [self.n_mus] + self.layers_mu
        self.steps = len(self.layers)-1

        if self.mode == 'standard' or self.mode == 'parametric':
            self.loss_names = ['total loss', 'image loss']
            self.idx_early_stop = 1
        elif self.mode == 'combined':
            self.loss_names = ['total loss', 'image nn loss', 'image mu loss', 'code loss']
            self.idx_early_stop = 2
        elif 'staggered' in self.mode:
            self.loss_names = ['total loss', 'image loss']  # TODO: change if we compute code loss instead of image loss
            self.idx_early_stop = 1
        else:
            raise NotImplementedError

        if args.reg:
            self.loss_names.append('reg loss')
        if args.bias_ord:
            self.loss_names.append('bias loss')

        if 'param_relu' in [self.act_out, self.act_hid, self.act_code]:
            self.param_relu = param_relu(self.alpha_relu)
            self.param_activation = True
        if 'param_sigmoid' in [self.act_out, self.act_hid, self.act_code]:
            self.param_sigmoid = param_sigmoid(self.alpha_sigmoid)
            self.param_activation = True
        
        # Build standard, combined or parametric
        if self.mode == 'standard':
            self.encoder = self.Encoder(self).to(torch.double)
            self.decoder = self.Decoder(self).to(torch.double)
            self.parameter = None
        elif self.mode == 'combined' or 'staggered' in self.mode:
            self.encoder = self.Encoder(self).to(torch.double)
            self.decoder = self.Decoder(self).to(torch.double)
            self.parameter = self.Parameter(self).to(torch.double)
        elif self.mode == 'parametric':
            self.encoder = None
            self.decoder = self.Decoder(self).to(torch.double)
            self.parameter = self.Parameter(self).to(torch.double)
        else:
            raise NotImplementedError

        self.models = [self.encoder, self.decoder, self.parameter]
        self.n_train_params, self.n_biases  = self.__count_params()
        self.__summary()

    def __count_params(self):
        n_train_params = 0
        n_biases = 0
        for model in self.models:
            if model:
                for p in model.parameters():
                    if p.requires_grad:
                        if len(p.shape) == 2: # p are weights
                            n_train_params += p.shape[0]*p.shape[1]
                        elif len(p.shape) == 1: # p are biases
                            n_train_params += p.shape[0]
                            n_biases += p.shape[0]
                        else:
                            raise ValueError
        return n_train_params, n_biases


    def __summary(self):
        name = f'{self.fig_path}/archTable_{self.name}.png'
        data = [['autoencoder mode', self.mode],
            ['encoder/decoder layers', self.layers],
            ['num train params', self.n_train_params],
            ['num biases', self.n_biases]]
        if self.mode == 'combined' or self.mode == 'parametric' or 'staggered' in self.mode:
            data.append(['parametric layers', self.layers_mu])
        
        data = data + [
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
        
        summaryInfo(data, name, self.verbose, self.save_fig)

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
        elif activation == 'tanh': x = torch.tanh(x) + 1
        elif activation == 'elu': x = F.elu(x)
        elif activation == 'prelu': x = torch.prelu(x, weight=torch.Tensor([0.1]).double())
        elif activation == 'param_sigmoid': x = self.param_sigmoid(x)
        elif activation == 'param_relu': x = self.param_relu(x)
        else: raise NotImplementedError
        return x

    class Encoder(nn.Module):
        def __init__(self, autoencoder):
            super().__init__()
            self.autoencoder = autoencoder
            self.name = 'encoder'
            self.build()

        def build(self):
            # Encoder
            self.encoder = nn.ModuleList()
            self.activation_encoder = []
            self.encoder.append(nn.Flatten())
            self.activation_encoder.append('linear')
            if self.autoencoder.dropout:
                self.encoder.append(nn.Dropout(p=self.autoencoder.dropout_prob))
                self.activation_encoder.append('linear')
            # Encoder hidden layers
            for k in range(self.autoencoder.steps-1):
                self.encoder.append(nn.Linear(self.autoencoder.layers[k], self.autoencoder.layers[k+1]))
                self.activation_encoder.append(self.autoencoder.act_hid)
                if self.autoencoder.dropout:
                    self.encoder.append(nn.Dropout(p=self.autoencoder.dropout_prob))
                    self.activation_encoder.append('linear')
            # Code
            self.encoder.append(nn.Linear(self.autoencoder.layers[-2], self.autoencoder.layers[-1]))
            self.activation_encoder.append(self.autoencoder.act_code)

            # Weight initialisation  
            self.autoencoder.weight_init(self.encoder)

        def encode(self, x):
            return self.autoencoder.layersLoop(x, self.encoder, self.activation_encoder)

        def forward(self, x):
            return self.encode(x)

    class Decoder(nn.Module):
        def __init__(self, autoencoder):
            super().__init__()
            self.autoencoder = autoencoder
            self.name = 'decoder'
            self.build()

        def build(self):
            # Decoder
            self.decoder = nn.ModuleList()
            self.activation_decoder = []
            self.decoder.append(nn.Flatten())
            self.activation_decoder.append('linear')
            for k in range(self.autoencoder.steps-1):
                self.decoder.append(nn.Linear(self.autoencoder.layers[self.autoencoder.steps-k], self.autoencoder.layers[self.autoencoder.steps-k-1]))
                self.activation_decoder.append(self.autoencoder.act_hid)
                if self.autoencoder.dropout:
                    self.decoder.append(nn.Dropout(p=self.autoencoder.dropout_prob))
                    self.activation_decoder.append('linear')
            # Add last decoder layer
            self.decoder.append(nn.Linear(self.autoencoder.layers[1], self.autoencoder.layers[0]))
            self.activation_decoder.append(self.autoencoder.act_out)

            # Weight initialisation  
            self.autoencoder.weight_init(self.decoder)

        def decode(self, x):
            return self.autoencoder.layersLoop(x, self.decoder, self.activation_decoder)

        def forward(self, x):
            return self.decode(x)


    class Parameter(nn.Module):
        def __init__(self, autoencoder):
            super().__init__()
            self.autoencoder = autoencoder
            self.name = 'parameter'
            self.build()

        def build(self):
            # Parameter
            steps_mu = len(self.autoencoder.layers_mu)-1
            self.parameter = nn.ModuleList()
            self.activation_param = []
            for k in range(steps_mu-1):
                self.parameter.append(nn.Linear(self.autoencoder.layers_mu[k], self.autoencoder.layers_mu[k+1]))
                self.activation_param.append(self.autoencoder.act_hid)
            self.parameter.append(nn.Linear(self.autoencoder.layers_mu[-2], self.autoencoder.layers_mu[-1]))
            self.activation_param.append(self.autoencoder.act_code)

            # Weight initialisation  
            self.autoencoder.weight_init(self.parameter)

        def param(self, x):
            return self.autoencoder.layersLoop(x, self.parameter, self.activation_param)

        def forward(self, x):
            return self.param(x)