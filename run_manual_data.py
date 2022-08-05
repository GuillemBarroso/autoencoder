from src.load_data import Data
from src.arch import Autoencoder
from src.train import Train
from src.predict import Predict
from src.losses import computeErrors
import matplotlib.pyplot as plt


class Input():
    def __init__(self):
        self.dataset = 'beam_homog'
        self.verbose = False
        self.plot = False
        self.save = False
        self.save_dir = 'models/manual_data'

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
        self.mode = 'parametric'
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
    nManualData = 12
    data_rng = range(nManualData)
    e_L1 = [None]*nManualData
    e_L2 = [None]*nManualData
    e_infty = [None]*nManualData

    for i in data_rng:
        args.manual_data = i
        out, data = run(args)

        ref = data.x_test[:,:,:,0]
        e_L1[i], e_L2[i], e_infty[i] = computeErrors(data.n_test, ref, out)
    
    fig, ax = plt.subplots()
    plt.grid(axis='x', color='0.9')
    plt.scatter(data_rng, e_L1)
    plt.scatter(data_rng, e_L2)
    plt.scatter(data_rng, e_infty)
    plt.ylabel('|in-out|_Lp / |in|_Lp')
    plt.xlabel("# manual data case")
    plt.ylim(0,1)
    ax.set_xticks(data_rng)
    ax.set_axisbelow(True)
    plt.legend(['p = 1', 'p = 2', 'p = infty'], loc='best')
    plt.show()



