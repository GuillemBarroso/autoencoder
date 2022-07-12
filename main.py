import argparse
from torchsummary import summary
from src.load_data import Data
from src.arch import Autoencoder
from src.model import Model
from src.predict import Predict


def main(args):
    # Load data
    data = Data(args)

    # Create autoencoder
    model = Autoencoder(data.resolution, args)
    if args.verbose:
        summary(model, data.resolution)

    # Train and predict
    Model(model, data, args).train()
    Predict(model, data, args).evaluate()    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder for image compression')

    # General parameters
    parser.add_argument('--dataset', '-d', default='ellipse2', type=str, help='name of the dataset')
    parser.add_argument('--verbose', '-vrb', default=True, type=bool, help='display information on command window')
    parser.add_argument('--plot', '-plt', default=True, type=bool, help='plot training and predictions in figures and save pngs')

    # Data parameters
    parser.add_argument('--random_test_data', '-rnd_data', default=True, type=bool, help='test data selected randomly (using "split_size"). If False, it will be loaded from test_data.py')
    parser.add_argument('--split_size', '-split_size', default=0.1, type=float, help='test and validation splitting percentage (from 0 to 1) from total dataset')
    
    # Training parameters
    parser.add_argument('--epochs', '-e', default=1000, type=int, help='number of training epochs')
    parser.add_argument('--reg', '-reg', default=True, type=bool, help='if True, adds a regularisation term in the loss function')
    parser.add_argument('--reg_coef', '-reg_coef', default=1e-4, type=float, help='coefficient that multiplies the regularisation term in the loss function')
    parser.add_argument('--batch_size', '-bs', default=600, type=int, help='batch size')
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float, help='training learning rate ')
    parser.add_argument('--early_stop_patience', '-es_pat', default=100, type=int, help='number of epochs that the early stopping criteria will wait before stopping training')
    parser.add_argument('--early_stop_tol', '-es_tol', default=1e-3, type=float, help='tolerance that the early stopping will consider')
    parser.add_argument('--lr_epoch_milestone', '-lr_e', default=[1000], nargs='+', type=int, help='list of epochs in which learning rate will be decreased')
    parser.add_argument('--lr_red_coef','-lr_coef', default=7e-1, type=float, help='learning rate reduction factor')

    # Architecture parameters
    parser.add_argument('--layers','-l', default=[200, 100, 25], nargs='+', type=int, help='list with number of neurons per layer (including code)')
    parser.add_argument('--initialisation','-init', default='kaiming_uniform', type=str, help='weight initialisation method')
    parser.add_argument('--act_code','-act_code', default='relu', type=str, help="activaction function in encoder's last layer (code or latent space)")
    parser.add_argument('--act_hid','-act_hid', default='param_relu', type=str, help="activaction function in autoencoder's hidden layers")
    parser.add_argument('--act_out','-act_out', default='param_sigmoid', type=str, help="activaction function in decoders's last layer (final output)")
    parser.add_argument('--alpha_relu','-a_relu', default=0.5, type=float, help="initial value for the parameter of the relu activaction function")
    parser.add_argument('--alpha_sigmoid','-a_sigmoid', default=0.5, type=float, help="initial value for the parameter of the sigmoid activaction function")
    parser.add_argument('--dropout','-drop', default=False, type=bool, help="Option that, if True, will activate dropout layers after each hidden layer")
    parser.add_argument('--dropout_prob','-drop_prob', default=0.1, type=float, help="Probability of each dropout layer to deactivate its neurons in a forward pass")

    # Display parameters
    parser.add_argument('--n_disp','-disp', default=6, type=int, help='number of test images displayed in results figure')

    args = parser.parse_args()

    main(args)

