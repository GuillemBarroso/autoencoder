import matplotlib.pyplot as plt

from src.load_data import Data
from src.arch import Autoencoder
from src.train import Train
from src.predict import Predict
from src.losses import computeErrors


class Input():
    def __init__(self):
        self.dataset = 'beam_homog'
        self.verbose = False
        self.plot = False
        self.save_model = False
        self.save_fig = False
        self.model_dir = f'models/manual_data'
        self.fig_dir = f'figures/manual_data'

        # Data parameters
        self.random_test_data = False
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

def run(args):
    # Load data
    data = Data(args)

    # Create autoencoder and load saved models
    autoencoder = Autoencoder(data, args)
    Train(autoencoder, data, args)

    pred = Predict(autoencoder, data, args)

    return pred.nn_out[0], data

if __name__ == "__main__":
    args = Input()
    n_cases = 12
    modes = ['standard', 'combined', 'parametric']
    data_rng = range(n_cases)

    e_std_L1 = [None]*n_cases
    e_std_L2 = [None]*n_cases
    e_std_infty = [None]*n_cases
    e_comb_L1 = [None]*n_cases
    e_comb_L2 = [None]*n_cases
    e_comb_infty = [None]*n_cases
    e_param_L1 = [None]*n_cases
    e_param_L2 = [None]*n_cases
    e_param_infty = [None]*n_cases

    for i in data_rng:
        for mode in modes:
            print(f'n_case: {i}/{n_cases}, mode: {mode}')
            args.mode = mode
            args.manual_data = i
            out, data = run(args)

            ref = data.x_test[:,:,:,0]
            e_L1, e_L2, e_infty = computeErrors(data.n_test, ref, out)
            # Store
            if mode == 'standard':
                e_std_L1[i] = e_L1*100
                e_std_L2[i] = e_L2*100
                e_std_infty[i] = e_infty*100
            elif mode == 'combined':
                e_comb_L1[i] = e_L1*100
                e_comb_L2[i] = e_L2*100
                e_comb_infty[i] = e_infty*100
            elif mode == 'parametric':
                e_param_L1[i] = e_L1*100
                e_param_L2[i] = e_L2*100
                e_param_infty[i] = e_infty*100

    # Load parametric averages from cross validation
    std_avg = [9.73, 4.8]
    comb_avg = [30.97, 14.43]
    param_avg = [32.12, 15.13]

    # Plot results
    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(data_rng, e_std_L1)
    plt.scatter(data_rng, e_comb_L1)
    plt.scatter(data_rng, e_param_L1)
    plt.ylabel('|in-out|_L1 / |in|_L1 [%]')
    plt.xlabel("# manual dataset case")
    plt.ylim(0,80)
    ax.set_xticks(data_rng)
    ax.set_axisbelow(True)
    plt.legend(['Standard', 'Combined', 'Parametric'], loc='upper left')
    plt.plot(data_rng, [std_avg[0]]*n_cases, '--r', zorder=0)
    plt.plot(data_rng, [comb_avg[0]]*n_cases, '--r', zorder=0)
    plt.plot(data_rng, [param_avg[0]]*n_cases, '--r', zorder=0)
    plt.text(1.1, std_avg[0] + 1, f"Standard avg = {std_avg[0]:.2f}", fontsize=9, color='r')
    plt.text(1.1, comb_avg[0] - 3, f"Combined avg = {comb_avg[0]:.2f}", fontsize=9, color='r')
    plt.text(1.1, param_avg[0] + 1, f"Parametric avg = {param_avg[0]:.2f}", fontsize=9, color='r')
    plt.savefig(f'{data.fig_path}/L1_{data.name}.png')

    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(data_rng, e_std_L2)
    plt.scatter(data_rng, e_comb_L2)
    plt.scatter(data_rng, e_param_L2)
    plt.ylabel('|in-out|_L2 / |in|_L2 [%]')
    plt.xlabel("# manual dataset case")
    plt.ylim(0,80)
    ax.set_xticks(data_rng)
    ax.set_axisbelow(True)
    plt.legend(['Standard', 'Combined', 'Parametric'], loc='upper left')
    plt.plot(data_rng, [std_avg[1]]*n_cases, '--r', zorder=0)
    plt.plot(data_rng, [comb_avg[1]]*n_cases, '--r', zorder=0)
    plt.plot(data_rng, [param_avg[1]]*n_cases, '--r', zorder=0)
    plt.text(1.1, std_avg[1] + 1, f"Standard avg = {std_avg[1]:.2f}", fontsize=9, color='r')
    plt.text(1.1, comb_avg[1] - 3, f"Combined avg = {comb_avg[1]:.2f}", fontsize=9, color='r')
    plt.text(1.1, param_avg[1] + 1, f"Parametric avg = {param_avg[1]:.2f}", fontsize=9, color='r')

    plt.savefig(f'{data.fig_path}/L2_{data.name}.png')
    plt.show()




