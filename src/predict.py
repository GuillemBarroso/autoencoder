import torch
from src.calcs import computeLosses, codeInfo
from src.postprocess import storeLossInfo, summaryInfo, plotting, plotShow, reshape, addLossesToList
import numpy as np

class Predict(object):
    def __init__(self, autoencoder, data, args):
        self.x_test = data.x_test
        self.n_test = data.n_test
        self.mus_test = data.mus_test
        self.resolution = data.resolution
        self.img_names_test = data.img_names_test
        self.data_class = data.data_class
        self.n_train_params = data.n_train_params
        self.reg = args.reg
        self.reg_coef = args.reg_coef
        self.code_size = args.layers[-1]
        if args.n_disp > len(self.x_test): args.n_disp = len(self.x_test)
        self.n_disp = args.n_disp
        self.verbose = args.verbose
        self.plot = args.plot
        self.mode = args.mode
        self.losses_weights = args.losses_weights

        self.autoencoder = autoencoder
        self.encoder = autoencoder[0]
        self.decoder = autoencoder[1]
        if self.mode == 'parametric':
            self.parameter = autoencoder[2]

        self.loss_test = [[] for x in range(len(self.encoder.loss_names))]
        self.zero_code_flag = None
        self.active_code_size = None
        self.avg_code_mag = None
        self.zero_code_flag_mu = None
        self.active_code_size_mu = None
        self.avg_code_mag_mu = None
        self.code_dim = (self.n_test, int(np.sqrt(self.code_size)), int(np.sqrt(self.code_size)))
        self.img_dim = (self.n_test, self.resolution[0], self.resolution[1])

    def evaluate(self):
        def __summary():
            name = 'results/evaluationTable.png'
            data = [
            ['code nn size/max size', '{}/{}'.format(self.active_code_size, self.code_size)],
            ['avg pixel magnitude nn', '{:.2}'.format(self.avg_code_mag)],
            ]
            if self.mode == 'parametric':
                data.append(['code mu size/max size', '{}/{}'.format(self.active_code_size_mu, self.code_size)])
                data.append(['avg pixel magnitude mu', '{:.2}'.format(self.avg_code_mag_mu)])

            data = addLossesToList(self.loss_test, 'test', self.encoder.loss_names, data)
            summaryInfo(data, name, self.verbose)

        with torch.no_grad():
            if self.mode == 'parametric':
                code_nn = self.encoder(self.x_test.data)
                code_mu = self.parameter(self.mus_test.data)
                X_nn = self.decoder(code_nn)
                X_mu = self.decoder(code_mu)

                # Reshape arrays for plotting
                code_nn = reshape(code_nn, self.code_dim)
                code_mu = reshape(code_mu, self.code_dim)
                X_nn = reshape(X_nn, self.img_dim)
                X_mu = reshape(X_mu, self.img_dim)

                out = [X_nn, X_mu, code_nn, code_mu]

            elif self.mode == 'standard':
                code_nn = self.encoder(self.x_test.data)
                X_nn = self.decoder(code_nn)

                # Reshape arrays for plotting
                code_nn = reshape(code_nn, self.code_dim)
                X_nn = reshape(X_nn, self.img_dim)
                out = [X_nn, code_nn]
            else:
                raise NotImplementedError

            loss = computeLosses(out, self.x_test.data, self.autoencoder, self.reg, self.reg_coef, self.mode, self.n_train_params, self.losses_weights)

        storeLossInfo(loss, self.loss_test)

        # Get information on both codes
        self.zero_code_flag, self.active_code_size, self.avg_code_mag = codeInfo(code_nn)
        zero_code = [self.zero_code_flag]

        if self.mode == 'parametric':
            self.zero_code_flag_mu, self.active_code_size_mu, self.avg_code_mag_mu = codeInfo(code_mu)
            zero_code += [self.zero_code_flag_mu]

        # Select first n_disp test images for display
        x_test = self.x_test[:self.n_disp,:,:]
        img_test = self.img_names_test[:self.n_disp]
        
        __summary()
        if self.plot:
            plotting(x_test, out, img_test, zero_code, self.data_class, self.mode)
            plotShow()

        
