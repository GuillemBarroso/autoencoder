import torch
from src.calcs import computeLosses, codeInfo, truncCode
from src.postprocess import summaryInfo, plotting, plotShow, reshape
import numpy as np

class Predict(object):
    def __init__(self, model, data, args):
        self.model = model
        self.x_test = data.x_test
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

        self.loss_tot = None
        self.loss_image = None
        self.loss_reg = None
        self.zero_code_flag = None
        self.trunc_code_flag = None

    def evaluate(self):
        def __summary():
            name = 'results/evaluationTable.png'
            data = [
            ['code size/max size', '{}/{}'.format(self.active_code_size, self.code_size)],
            ['avg pixel magnitude', '{:.2}'.format(self.avg_code_mag)],
            ['total test loss', '{:.2}'.format(self.loss_tot)],
            ['image test loss', '{:.2}'.format(self.loss_image)],
            ['reg test loss', '{:.2}'.format(self.loss_reg)],
            ]
            summaryInfo(data, name, self.verbose)

        with torch.no_grad():
            pred, code = self.model(self.x_test.data)
            self.loss_tot, self.loss_image, self.loss_reg = computeLosses(pred, self.x_test.data, self.model, self.reg, self.reg_coef)

        self.zero_code_flag, self.active_code_size, self.avg_code_mag = codeInfo(code)

        # Select first n_disp test images for display
        x_test = self.x_test[:self.n_disp,:,:]
        img_test = self.img_names[:self.n_disp]
        code_dim = (code.shape[0], int(np.sqrt(self.code_size)), int(np.sqrt(self.code_size)))
        pred_dim = (pred.shape[0], self.resolution[0], self.resolution[1])
        code = reshape(code, code_dim)
        pred = reshape(pred, pred_dim)
        
        __summary()
        if self.plot:
            plotting(x_test, code, pred, img_test, self.zero_code_flag, self.trunc_code_flag, self.data_class)
            plotShow()

        
