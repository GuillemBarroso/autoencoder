import torch
from src.calcs import computeLosses, codeInfo, truncCode
from src.postprocess import summaryInfo, plotting, plotShow, reshape
import numpy as np

class Predict(object):
    def __init__(self, model, x_test, img_names, n_disp, args):
        self.model = model
        self.x_test = x_test
        self.img_names = img_names
        self.reg_coef = args.reg_coef
        self.code_size = args.code_size
        if n_disp > len(x_test): n_disp = len(x_test)
        self.n_disp = n_disp

        self.loss_tot = None
        self.loss_image = None
        self.loss_reg = None
        self.latent_trunc_size = None
        self.loss_tot_trunc = None
        self.loss_image_trunc = None
        self.loss_reg_trunc = None

    def evaluate(self):
        def __summary():
            name = 'results/evaluationTable.png'
            data = [
            ['code size/max size', '{}/{}'.format(self.active_code_size, self.code_size)],
            ['avg pixel magnitude', '{:.2}'.format(self.avg_code_mag)],
            ['total test loss', '{:.2}'.format(self.loss_tot)],
            ['image test loss', '{:.2}'.format(self.loss_image)],
            ['reg test loss', '{:.2}'.format(self.loss_reg)],
            ['----------', '----------'],
            ['trunc threshold', '{}'.format(0.1)],
            ['trunc code size', '{}'.format(self.latent_trunc_size)],
            ['total trunc test loss', '{:.2}'.format(self.loss_tot_trunc)],
            ['image trunc test loss', '{:.2}'.format(self.loss_image_trunc)],
            ['reg trunc test loss', '{:.2}'.format(self.loss_reg_trunc)],
            ]
            summaryInfo(data, name)

        with torch.no_grad():
            pred, code = self.model(self.x_test.data)
            self.loss_tot, self.loss_image, self.loss_reg = computeLosses(pred, self.x_test.data, code, self.reg_coef)

        active_code, self.active_code_size, self.avg_code_mag = codeInfo(code)
        code_trunc, self.latent_trunc_size  = truncCode(code, self.code_size)
        print('Code for test image 1: ', code[0])
        print('Total test loss before truncation: ',self.loss_tot)
        print('Truncated code for test image 1: ',code_trunc[0])
        with torch.no_grad():
            pred_trunc = self.model.decode(code_trunc)
            self.loss_tot_trunc, self.loss_image_trunc, self.loss_reg_trunc = computeLosses(pred_trunc, self.x_test.data, code_trunc, self.reg_coef)
        print('Total test loss after truncation: ',self.loss_tot_trunc)
        print('Prediction from code - prediciton from truncarted code:')
        print(pred[0]-pred_trunc[0])

        # Select first n_disp test images for display
        x_test = self.x_test[:self.n_disp,:,:]
        img_test = self.img_names[:self.n_disp]
        codeSize = (code.shape[0], int(np.sqrt(self.code_size)), int(np.sqrt(self.code_size)))
        predSize = (pred.shape[0], self.x_test.resolution[0], self.x_test.resolution[1])
        code = reshape(code, codeSize)
        code_trunc = reshape(code_trunc, codeSize)
        pred = reshape(pred, predSize)
        pred_trunc = reshape(pred_trunc, predSize)
        
        __summary()
        plotting(x_test, code, pred, code_trunc, pred_trunc, img_test)
        plotShow()

        
