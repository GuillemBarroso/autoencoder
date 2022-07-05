import argparse
from torchsummary import summary
from src.load_data import Data, DataTorch
from src.arch import Autoencoder
from src.model import Model
from src.predict import Predict


def main(args):
    data = Data(args.dataset)
    x_train = DataTorch(data.x_train)
    x_val = DataTorch(data.x_val)
    x_test = DataTorch(data.x_test)

    model = Autoencoder(data.resolution, args)
    if args.verbose:
        summary(model, data.resolution)

    Model(model, x_train, x_val, args).train()
    Predict(model, x_test, data.imgTestNames, args.n_disp, args).evaluate()    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder for image compression')

    # General parameters
    parser.add_argument('--dataset', '-d', default='beam_homog_test', type=str, help='name of the dataset')
    parser.add_argument('--trunc_threshold', '-th', default=0.1, type=float, help='threshold to truncate the code after training')
    parser.add_argument('--verbose', '-v', default=True, type=bool, help='display information on command window')
    parser.add_argument('--plot', '-p', default=True, type=bool, help='plot training and predictions in figures and save pngs')

    # Training parameters
    parser.add_argument('--epochs', '-e', default=1000, type=int, help='number of training epochs')
    parser.add_argument('--reg_coef', '-reg', default=1e-4, type=float, help='regularisation coefficient in the code layer')
    parser.add_argument('--batch_size', '-bs', default=50, type=int, help='batch size')
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float, help='training learning rate ')
    parser.add_argument('--early_stop_patience', '-pat', default=50, type=int, help='number of epochs that the early stopping criteria will wait before stopping training')
    parser.add_argument('--early_stop_tol', '-tol', default=1e-3, type=float, help='tolerance that the early stopping will consider')
    parser.add_argument('--epoch_milestone', '-ered', default=[1e10], nargs='+', type=int, help='list of epochs in which learning rate will be decreased')
    parser.add_argument('--lr_red_coef','-lrred', default=1e-1, type=float, help='learning rate reduction factor')

    # Architecture parameters
    parser.add_argument('--layers','-l', default=[200, 200, 25], nargs='+', type=int, help='list with number of neurons per layer (including code)')
    parser.add_argument('--activation','-act', default='relu', type=str, help="activaction function in autoencoder's hidden layers")
    parser.add_argument('--initialisation','-init', default='kaiming_uniform', type=str, help='weight initialisation method')

    # Display parameters
    parser.add_argument('--n_disp','-disp', default=6, type=int, help='number of test images displayed in results figure')

    args = parser.parse_args()

    main(args)

