import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class param_sigmoid(nn.Module):
    def __init__(self, alpha = None):
        super(param_sigmoid,self).__init__()
        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(1.0))
        else:
            self.alpha = Parameter(torch.tensor(alpha))
            
        self.alpha.requiresGrad = True

    def forward(self, x):
        return 1 / (1+torch.exp(-self.alpha*x))


class param_relu(nn.Module):
    def __init__(self, alpha = None):
        super(param_relu,self).__init__()

        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(1.0))
        else:
            self.alpha = Parameter(torch.tensor(alpha))
            
        self.alpha.requiresGrad = True

    def forward(self, x):
        return torch.max(torch.tensor(0), self.alpha*x)