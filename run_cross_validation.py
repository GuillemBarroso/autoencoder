import matplotlib.pyplot as plt
import numpy as np

from src.load_data import Data
from src.arch import Autoencoder
from src.train import Train
from src.predict import Predict
from src.losses import computeErrors

class Input():
    def __init__(self):
        self.dataset = 'beam_homog'
        self.verbose = True
        self.plot = True
        self.save_model = False
        self.save_model = True
        self.save_fig = True
        self.model_dir = 'models/cross_validation'
        self.fig_dir = 'figures/cross_validation'

        # Data parameters
        self.random_test_data = True
        self.random_seed = 1
        self.split_size = 0.1
        self.manual_data = 0

        # Training parameters
        self.epochs = 5000
        self.batch_size = 600
        self.learning_rate = 1e-3
        self.reg = True
        self.reg_coef = 1e-4
        self.code_coef = 1e-3
        self.bias_ord = False
        self.bias_coef = 1

        self.early_stop_patience = 1000
        self.early_stop_tol = 0.1

        self.lr_epoch_milestone = [5000]
        self.lr_red_coef = 7e-1

        # Architecture parameters
        self.mode = 'standard'
        self.layers = [200, 100, 25]
        self.layers_mu = [200, 200, 200, 25]
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


def validation(args):
    # Load data
    data = Data(args)

    # Create autoencoder and load saved models
    autoencoder = Autoencoder(data, args)
    Train(autoencoder, data, args)

    pred = Predict(autoencoder, data, args)

    return pred.nn_out[0], data

if __name__ == "__main__":

    args = Input()

    n_seeds = 5
    modes = ['standard', 'combined', 'parametric', 'staggered_img', 'staggered_code']
    e_std_L1 = [None]*n_seeds
    e_std_L2 = [None]*n_seeds
    e_comb_L1 = [None]*n_seeds
    e_comb_L2 = [None]*n_seeds
    e_param_L1 = [None]*n_seeds
    e_param_L2 = [None]*n_seeds
    e_stag_img_L1 = [None]*n_seeds
    e_stag_img_L2 = [None]*n_seeds
    e_stag_code_L1 = [None]*n_seeds
    e_stag_code_L2 = [None]*n_seeds
    
    seeds_rng = range(1, n_seeds+1)
    for seed in seeds_rng:
        for mode in modes:
            print(f'n_case: {seed}/{n_seeds}, mode: {mode}')
            args.mode = mode
            args.random_seed = seed
            out, data = validation(args)

            e_L1, e_L2, _ = computeErrors(data.n_test, data.x_test[:,:,:,0], out)

            # Store
            if mode == 'standard':
                e_std_L1[seed-1] = e_L1*100
                e_std_L2[seed-1] = e_L2*100
            elif mode == 'combined':
                e_comb_L1[seed-1] = e_L1*100
                e_comb_L2[seed-1] = e_L2*100
            elif mode == 'parametric':
                e_param_L1[seed-1] = e_L1*100
                e_param_L2[seed-1] = e_L2*100
            elif mode == 'staggered_img':
                e_stag_img_L1 = e_L1*100
                e_stag_img_L2 = e_L2*100
            elif mode == 'staggered_code':
                e_stag_code_L1 = e_L1*100
                e_stag_code_L2 = e_L2*100
    
    # Compute average between all seeds
    e_std_L1_avg = np.average(e_std_L1)
    e_std_L2_avg = np.average(e_std_L2)
    e_comb_L1_avg = np.average(e_comb_L1)
    e_comb_L2_avg = np.average(e_comb_L2)
    e_param_L1_avg = np.average(e_param_L1)
    e_param_L2_avg = np.average(e_param_L2)
    e_stag_img_L1_avg = np.average(e_stag_img_L1)
    e_stag_img_L2_avg = np.average(e_stag_img_L2)
    e_stag_code_L1_avg = np.average(e_stag_code_L1)
    e_stag_code_L2_avg = np.average(e_stag_code_L2)

    # Plot results
    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(seeds_rng, e_std_L1)
    plt.scatter(seeds_rng, e_comb_L1)
    plt.scatter(seeds_rng, e_param_L1)
    plt.scatter(seeds_rng, e_stag_img_L1)
    plt.scatter(seeds_rng, e_stag_code_L1)
    plt.ylabel('|in-out|_L1 / |in|_L1 [%]')
    plt.xlabel("# dataset's seed")
    plt.ylim(0,40)
    ax.set_xticks(seeds_rng)
    ax.set_axisbelow(True)
    plt.legend(['Standard', 'Combined', 'Parametric', 'Staggered img', 'Staggered code'], loc='upper right')
    plt.plot(seeds_rng, [e_std_L1_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_comb_L1_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_param_L1_avg]*n_seeds, '--r', zorder=0)
    plt.text(1.1, e_std_L1_avg + 1, f"Standard avg = {e_std_L1_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_comb_L1_avg - 2, f"Combined avg = {e_comb_L1_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_param_L1_avg + 1, f"Parametric avg = {e_param_L1_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_param_L1_avg + 1, f"Staggered img avg = {e_stag_img_L1_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_param_L1_avg + 1, f"Staggered code avg = {e_stag_code_L1_avg:.2f}", fontsize=9, color='r')
    plt.savefig(f'{data.fig_path}/L1_{data.name}.png')

    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(seeds_rng, e_std_L2)
    plt.scatter(seeds_rng, e_comb_L2)
    plt.scatter(seeds_rng, e_param_L2)
    plt.scatter(seeds_rng, e_stag_img_L2)
    plt.scatter(seeds_rng, e_stag_code_L2)
    plt.ylabel('|in-out|_L2 / |in|_L2 [%]')
    plt.xlabel("# dataset's seed")
    plt.ylim(0,40)
    ax.set_xticks(seeds_rng)
    ax.set_axisbelow(True)
    plt.legend(['Standard', 'Combined', 'Parametric', 'Staggered img', 'Staggered code'], loc='upper right')
    plt.plot(seeds_rng, [e_std_L2_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_comb_L2_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_param_L2_avg]*n_seeds, '--r', zorder=0)
    plt.text(1.1, e_std_L2_avg + 1, f"Standard avg = {e_std_L2_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_comb_L2_avg - 2, f"Combined avg = {e_comb_L2_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_param_L2_avg + 1, f"Parametric avg = {e_param_L2_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_param_L2_avg + 1, f"Staggered img avg = {e_stag_img_L2_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_param_L2_avg + 1, f"Staggered code avg = {e_stag_code_L2_avg:.2f}", fontsize=9, color='r')
    plt.savefig(f'{data.fig_path}/L2_{data.name}.png')

    plt.show()
