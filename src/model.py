from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, data, args):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(data.resolution[0]*data.resolution[1], args.n_neurons),
            nn.ReLU(),
            nn.Linear(args.n_neurons, args.n_neurons),
            nn.ReLU(),
            # nn.Linear(self.nNeurons, self.nNeurons),
            # nn.ReLU(),
            # nn.Linear(self.nNeurons, self.nNeurons),
            # nn.ReLU(),
            nn.Linear(args.n_neurons, args.code_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(args.code_size, args.n_neurons),
            nn.ReLU(),
            nn.Linear(args.n_neurons, args.n_neurons),
            nn.ReLU(),
            # nn.Linear(self.nNeurons, self.nNeurons),
            # nn.ReLU(),
            # nn.Linear(self.nNeurons, self.nNeurons),
            # nn.ReLU(),
            nn.Linear(args.n_neurons, data.resolution[0]*data.resolution[1]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        code = self.encoder(x)
        x_nn = self.decoder(code)
        return x_nn, code