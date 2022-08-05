from src.load_data import Data
from src.arch import Autoencoder
from src.train import Train
from src.predict import Predict
from src.losses import computeErrors
import matplotlib.pyplot as plt
import numpy as np

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

    return pred.nn_out[0]

if __name__ == "__main__":

    args = Input()
    data = Data(args)
    ref = data.x_test[:,:,:,0]

    n_seeds = 5
    modes = ['standard', 'combined', 'parametric']
    e_std_L1 = [None]*n_seeds
    e_std_L2 = [None]*n_seeds
    e_std_infty = [None]*n_seeds
    e_comb_L1 = [None]*n_seeds
    e_comb_L2 = [None]*n_seeds
    e_comb_infty = [None]*n_seeds
    e_param_L1 = [None]*n_seeds
    e_param_L2 = [None]*n_seeds
    e_param_infty = [None]*n_seeds
    
    seeds_rng = range(1, n_seeds+1)
    for seed in seeds_rng:
        for mode in modes:
            args.mode = mode
            args.random_seed = seed
            out = validation(data, args)

            e_L1, e_L2, e_infty = computeErrors(data.n_test, ref, out)

            # Store
            if mode == 'standard':
                e_std_L1[seed-1] = e_L1
                e_std_L2[seed-1] = e_L2
                e_std_infty[seed-1] = e_infty
            elif mode == 'combined':
                e_comb_L1[seed-1] = e_L1
                e_comb_L2[seed-1] = e_L2
                e_comb_infty[seed-1] = e_infty
            elif mode == 'parametric':
                e_param_L1[seed-1] = e_L1
                e_param_L2[seed-1] = e_L2
                e_param_infty[seed-1] = e_infty
    
    # Compute average between all seeds
    e_std_L1_avg = np.average(e_std_L1)
    e_std_L2_avg = np.average(e_std_L2)
    e_std_infty_avg = np.average(e_std_infty)
    e_comb_L1_avg = np.average(e_comb_L1)
    e_comb_L2_avg = np.average(e_comb_L2)
    e_comb_infty_avg = np.average(e_comb_infty)
    e_param_L1_avg = np.average(e_param_L1)
    e_param_L2_avg = np.average(e_param_L2)
    e_param_infty_avg = np.average(e_param_infty)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot results
    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(seeds_rng, e_std_L1)
    plt.scatter(seeds_rng, e_comb_L1)
    plt.scatter(seeds_rng, e_param_L1)
    plt.ylabel('|in-out|_L1 / |in|_L1')
    plt.xlabel("# dataset's seed")
    plt.ylim(0,1)
    ax.set_xticks(seeds_rng)
    ax.set_axisbelow(True)
    plt.legend(['Standard', 'Combined', 'Parametric'], loc='upper right')
    plt.plot(seeds_rng, [e_std_L1_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_comb_L1_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_param_L1_avg]*n_seeds, '--r', zorder=0)
    plt.text(1.1, e_std_L1_avg + 0.02, f"Standard avg = {e_std_L1_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_comb_L1_avg - 0.04, f"Combined avg = {e_comb_L1_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_param_L1_avg + 0.02, f"Parametric avg = {e_param_L1_avg:.2f}", fontsize=9, color='r')

    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(seeds_rng, e_std_L2)
    plt.scatter(seeds_rng, e_comb_L2)
    plt.scatter(seeds_rng, e_param_L2)
    plt.ylabel('|in-out|_L2 / |in|_L2')
    plt.xlabel("# dataset's seed")
    plt.ylim(0,1)
    ax.set_xticks(seeds_rng)
    ax.set_axisbelow(True)
    plt.legend(['Standard', 'Combined', 'Parametric'], loc='upper right')
    plt.plot(seeds_rng, [e_std_L2_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_comb_L2_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_param_L2_avg]*n_seeds, '--r', zorder=0)
    plt.text(1.1, e_std_L2_avg + 0.02, f"Standard avg = {e_std_L2_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_comb_L2_avg - 0.04, f"Combined avg = {e_comb_L2_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_param_L2_avg + 0.02, f"Parametric avg = {e_param_L2_avg:.2f}", fontsize=9, color='r')

    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(seeds_rng, e_std_infty)
    plt.scatter(seeds_rng, e_comb_infty)
    plt.scatter(seeds_rng, e_param_infty)
    plt.ylabel('|in-out|_Linfty / |in|_Linfty')
    plt.xlabel("# dataset's seed")
    plt.ylim(0,1)
    ax.set_xticks(seeds_rng)
    ax.set_axisbelow(True)
    plt.legend(['Standard', 'Combined', 'Parametric'], loc='lower right')
    plt.plot(seeds_rng, [e_std_infty_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_comb_infty_avg]*n_seeds, '--r', zorder=0)
    plt.plot(seeds_rng, [e_param_infty_avg]*n_seeds, '--r', zorder=0)
    plt.text(1.1, e_std_infty_avg - 0.05, f"Standard avg = {e_std_infty_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_comb_infty_avg - 0.09, f"Combined avg = {e_comb_infty_avg:.2f}", fontsize=9, color='r')
    plt.text(1.1, e_param_infty_avg - 0.14, f"Parametric avg = {e_param_infty_avg:.2f}", fontsize=9, color='r')

    plt.show()
