import torch
from src.calcs_mu import computeLosses, codeInfo
from src.postprocess_mu import storeLossInfo, summaryInfo, plotting, plotShow, reshape, addLossesToList
import numpy as np

class Predict(object):
    def __init__(self, model, data, args):
        self.model = model
        self.x_test = data.x_test
        self.mus_test = data.mus_test
        self.resolution = data.resolution
        self.img_names = data.img_names
        self.data_class = data.data_class
        self.reg = args.reg
        self.reg_coef = args.reg_coef
        self.code_size = args.layers[-1]
        if args.n_disp > len(self.x_test): args.n_disp = len(self.x_test)
        self.n_disp = args.n_disp
        self.verbose = args.verbose
        self.plot = args.plot

        self.loss_test = [[] for x in range(len(self.model.loss_names))]
        self.zero_code_flag = None
        self.active_code_size = None
        self.avg_code_mag = None
        self.zero_code_flag_mu = None
        self.active_code_size_mu = None
        self.avg_code_mag_mu = None

    def evaluate(self):
        def __summary():
            name = 'results/evaluationTable.png'
            data = [
            ['code nn size/max size', '{}/{}'.format(self.active_code_size, self.code_size)],
            ['avg pixel magnitude nn', '{:.2}'.format(self.avg_code_mag)],
            ['code mu size/max size', '{}/{}'.format(self.active_code_size_mu, self.code_size)],
            ['avg pixel magnitude mu', '{:.2}'.format(self.avg_code_mag_mu)],
            ]
            data = addLossesToList(self.loss_test, 'test', self.model.loss_names, data)
            summaryInfo(data, name, self.verbose)

        with torch.no_grad():
            out = self.model(self.x_test.data, self.mus_test)
        loss = computeLosses(out, self.x_test.data, self.model, self.reg, self.reg_coef)
        storeLossInfo(loss, self.loss_test)

        # Get information on both codes
        X_nn = out[0]; X_mu = out[1]; code_nn = out[2]; code_mu = out[3]
        self.zero_code_flag, self.active_code_size, self.avg_code_mag = codeInfo(code_nn)
        self.zero_code_flag_mu, self.active_code_size_mu, self.avg_code_mag_mu = codeInfo(code_mu)

        # Select first n_disp test images for display
        x_test = self.x_test[:self.n_disp,:,:]
        img_test = self.img_names[:self.n_disp]
        code_dim = (code_nn.shape[0], int(np.sqrt(self.code_size)), int(np.sqrt(self.code_size)))
        X_dim = (X_nn.shape[0], self.resolution[0], self.resolution[1])
        code_nn = reshape(code_nn, code_dim)
        code_mu = reshape(code_mu, code_dim)
        X_nn = reshape(X_nn, X_dim)
        X_mu = reshape(X_mu, X_dim)
        
        __summary()
        if self.plot:
            plotting(x_test, code_nn, code_mu, X_nn, X_mu, img_test, self.zero_code_flag, self.zero_code_flag_mu, self.data_class)
            plotShow()

        
