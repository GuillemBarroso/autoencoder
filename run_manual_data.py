import matplotlib.pyplot as plt

from main import main
from src.losses import computeErrors


class Input():
    def __init__(self):
        self.dataset = 'beam_homog'
        self.verbose = False
        self.plot = False
        self.plot_show = False
        self.save_model = False
        self.save_fig = False
        self.model_dir = 'models/manual_data'
        self.fig_dir = 'figures/manual_data'

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

def getPlotName(epochs, reg, reg_coef, code_coef, layers, layers_mu):
    name = f"_epochs{epochs}"
    if reg:
        name += f"_regCoef{reg_coef}"
    name += f"_codeCoef{code_coef}_archED"
    for x in layers:
        name += f"_{x}"
    name += f"_archP"
    for x in layers_mu:
        name += f"_{x}"
    return name

if __name__ == "__main__":
    args = Input()
    n_cases = 12
    modes = ['standard', 'combined', 'parametric', 'staggered_img', 'staggered_code']
    data_rng = range(n_cases)

    e_std_L1 = [None]*n_cases
    e_std_L2 = [None]*n_cases
    e_comb_L1 = [None]*n_cases
    e_comb_L2 = [None]*n_cases
    e_param_L1 = [None]*n_cases
    e_param_L2 = [None]*n_cases
    e_stag_img_L1 = [None]*n_cases
    e_stag_img_L2 = [None]*n_cases
    e_stag_code_L1 = [None]*n_cases
    e_stag_code_L2 = [None]*n_cases

    for i in data_rng:
        for mode in modes:
            print(f'n_case: {i}/{n_cases}, mode: {mode}')
            args.mode = mode
            args.manual_data = i
            out, data = main(args)
            e_L1, e_L2, _ = computeErrors(data.n_test, data.x_test[:,:,:,0], out)

            # Store
            if mode == 'standard':
                e_std_L1[i] = e_L1*100
                e_std_L2[i] = e_L2*100
            elif mode == 'combined':
                e_comb_L1[i] = e_L1*100
                e_comb_L2[i] = e_L2*100
            elif mode == 'parametric':
                e_param_L1[i] = e_L1*100
                e_param_L2[i] = e_L2*100
            elif mode == 'staggered_img':
                e_stag_img_L1[i] = e_L1*100
                e_stag_img_L2[i] = e_L2*100
            elif mode == 'staggered_code':
                e_stag_code_L1[i] = e_L1*100
                e_stag_code_L2[i] = e_L2*100

    # Load parametric averages from cross validation
    L2_std_avg = [5.91, 5.91]
    L2_comb_avg = [15.00, 12.53]
    L2_param_avg = [12.92, 41.78]
    L2_stag_img_avg = [34.49, 51.43]
    L2_stag_code_avg = [63.51, 16.38]

    if args.layers_mu == [200, 200, 200, 25]:
        avg_idx = 1
    elif args.layers_mu == [50, 25]:
        avg_idx = 0

    # Plot results
    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(data_rng, e_std_L1)
    plt.scatter(data_rng, e_comb_L1)
    plt.scatter(data_rng, e_param_L1)
    plt.scatter(data_rng, e_stag_img_L1)
    plt.scatter(data_rng, e_stag_code_L1)
    plt.ylabel('|in-out|_L1 / |in|_L1 [%]')
    plt.xlabel("# manual dataset case")
    plt.ylim(0,120)
    ax.set_xticks(data_rng)
    ax.set_axisbelow(True)
    plt.legend(['Standard', 'Combined', 'Parametric', 'Staggered img', 'Staggered code'], loc='upper left')
    # plt.plot(data_rng, [std_avg[0]]*n_cases, '--r', zorder=0)
    # plt.plot(data_rng, [comb_avg[0]]*n_cases, '--r', zorder=0)
    # plt.plot(data_rng, [param_avg[0]]*n_cases, '--r', zorder=0)
    # plt.text(1.1, std_avg[0] + 1, f"Standard avg = {std_avg[0]:.2f}", fontsize=9, color='r')
    # plt.text(1.1, comb_avg[0] - 3, f"Combined avg = {comb_avg[0]:.2f}", fontsize=9, color='r')
    # plt.text(1.1, param_avg[0] + 1, f"Parametric avg = {param_avg[0]:.2f}", fontsize=9, color='r')
    plt.savefig(f'{data.fig_path}/L1_{getPlotName(args.epochs, args.reg, args.reg_coef, args.code_coef, args.layers, args.layers_mu)}.png')

    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(data_rng, e_std_L2)
    plt.scatter(data_rng, e_comb_L2)
    plt.scatter(data_rng, e_param_L2)
    plt.scatter(data_rng, e_stag_img_L2)
    plt.scatter(data_rng, e_stag_code_L2)
    plt.ylabel('|in-out|_L2 / |in|_L2 [%]')
    plt.xlabel("# manual dataset case")
    plt.ylim(0,80)
    ax.set_xticks(data_rng)
    ax.set_axisbelow(True)
    plt.legend(['Standard', 'Combined', 'Parametric', 'Staggered img', 'Staggered code'], loc='upper left')
    plt.plot(data_rng, [L2_std_avg[avg_idx]]*n_cases, '--r', zorder=0)
    plt.plot(data_rng, [L2_comb_avg[avg_idx]]*n_cases, '--r', zorder=0)
    plt.plot(data_rng, [L2_param_avg[avg_idx]]*n_cases, '--r', zorder=0)
    plt.plot(data_rng, [L2_stag_img_avg[avg_idx]]*n_cases, '--r', zorder=0)
    plt.plot(data_rng, [L2_stag_code_avg[avg_idx]]*n_cases, '--r', zorder=0)
    plt.text(1.1, L2_std_avg[avg_idx] + 1, f"Standard avg = {L2_std_avg[avg_idx]:.2f}", fontsize=9, color='r')
    plt.text(1.1, L2_comb_avg[avg_idx] - 3, f"Combined avg = {L2_comb_avg[avg_idx]:.2f}", fontsize=9, color='r')
    plt.text(1.1, L2_param_avg[avg_idx] + 1, f"Parametric avg = {L2_param_avg[avg_idx]:.2f}", fontsize=9, color='r')
    plt.text(1.1, L2_stag_img_avg[avg_idx] + 1, f"Staggered img avg = {L2_stag_img_avg[avg_idx]:.2f}", fontsize=9, color='r')
    plt.text(1.1, L2_stag_code_avg[avg_idx] + 1, f"Staggered code avg = {L2_stag_code_avg[avg_idx]:.2f}", fontsize=9, color='r')

    plt.savefig(f'{data.fig_path}/L2_{getPlotName(args.epochs, args.reg, args.reg_coef, args.code_coef, args.layers, args.layers_mu)}.png')
    plt.show()




