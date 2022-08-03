from src.load_data import Data
from src.arch import Autoencoder
from src.train import Train
from src.predict import Predict
import pandas as pd
import dataframe_image as dfi


class Input():
    def __init__(self):
        self.dataset = 'beam_homog'
        self.verbose = False
        self.plot = False
        self.save = False
        self.save_dir = 'models/cross_validation'

        # Data parameters
        self.random_test_data = True
        self.random_seed = 1
        self.split_size = 0.1
        self.manual_data = 0
        
        # Training parameters
        self.epochs = 2000
        self.batch_size = 600
        self.learning_rate = 1e-3
        self.reg = True
        self.reg_coef = 1e-4
        self.code_coef = 1e-3
        self.bias_ord = False
        self.bias_coef = 1
        
        self.early_stop_patience = 500
        self.early_stop_tol = 0.1
        
        self.lr_epoch_milestone = [5000]
        self.lr_red_coef = 7e-1

        # Architecture parameters
        self.mode = 'standard'
        self.layers = [200, 100, 25]
        self.layers_mu = [50, 25]
        self.initialisation = 'kaiming_uniform'
        
        self.act_code = 'relu'
        self.act_hid = 'relu'
        self.act_out = 'sigmoid'
        self.alpha_relu = 0.5
        self.alpha_sigmoid = 0.5
        
        self.dropout = False
        self.dropout_prob = 0.1

        # Display parameters
        self.n_disp = 6


def validation(data, args):

    # Create autoencoder and load saved models
    autoencoder = Autoencoder(data, args)
    Train(autoencoder, data, args)

    pred = Predict(autoencoder, data, args)

    return pred.loss_test

if __name__ == "__main__":

    args = Input()
    data = Data(args)

    nSeeds = 5
    modes = ['standard', 'combined', 'parametric']
    idx_loss = [1, 2, 1]
    loss_std = [None]*nSeeds
    loss_comb = [None]*nSeeds
    loss_param = [None]*nSeeds
    
    for seed in range(1, nSeeds+1):
        for iMode, mode in enumerate(modes):
            args.mode = mode
            args.random_seed = seed
            loss_model = validation(data, args)

            # Store
            if mode == 'standard':
                loss_std[seed-1] = loss_model[idx_loss[iMode]]
            elif mode == 'combined':
                loss_comb[seed-1] = loss_model[idx_loss[iMode]]
            elif mode == 'parametric':
                loss_param[seed-1] = loss_model[idx_loss[iMode]]
    loss = list(map(list, zip(*[loss_std, loss_comb, loss_param])))

    # Compute relative error with respect to standard method
    for seed in range(nSeeds):
        loss[seed].append(abs(loss[seed][0][0]-loss[seed][1][0])/loss[seed][0][0])
        loss[seed].append(abs(loss[seed][0][0]-loss[seed][2][0])/loss[seed][0][0])

    df = pd.DataFrame(loss, columns=['Standard', 'Combined', 'Parametric', 'Rel Error comb', 'Rel error param'])
    print(df)
    dfi.export(df, 'models/cross_validation/losses.png')
