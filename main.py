import torch
import argparse
import numpy as np
from torchsummary import summary
from src.load_data import Data, DataTorch
from arch import Autoencoder
from model import Model
from src.postprocess import plotting, plotShow


def main(args):
    data = Data(args.dataset)
    x_train = DataTorch(data.x_train)
    x_val = DataTorch(data.x_val)
    x_test = DataTorch(data.x_test)

    model = Autoencoder(data, args)
    summary(model, data.resolution)

    Model(model, x_train, x_val, x_test, args).train()
    
    # Make predictions
    n_disp = 6
    if n_disp > len(x_test): n_disp = len(x_test)
    x_test = x_test[:n_disp,:,:]
    img_test = data.imgTestNames[:n_disp]

    with torch.no_grad():
        pred, code = model(x_test)

    # Detect dimensionality
    latent = np.sum(code.numpy()**2, axis=0)**0.5
    rel_latent = latent/np.max(latent)

    active_code = np.zeros(args.code_size, dtype=int)

    for i in range(args.code_size):
        if rel_latent[i] > 0.1: active_code[i] = 1

    latent_trunc_dim = sum(active_code)

    code = torch.reshape(code, (code.shape[0], int(np.sqrt(args.code_size)), int(np.sqrt(args.code_size))))
    pred = torch.reshape(pred, (pred.shape[0], data.resolution[0], data.resolution[1]))
    
    plotting(x_test, code, pred, img_test)
    plotShow()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder for image compression')

    # General parameters
    parser.add_argument('--dataset', default='beam_homog_test', type=str, help='name of the dataset')

    # Training parameters
    parser.add_argument('--epochs', default=500, type=int, help='number of training epochs')
    parser.add_argument('--reg_coef', default=1e-4, type=float, help='regularisation coefficient in the code layer')
    parser.add_argument('--batch_size', default=50, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='training learning rate ')
    parser.add_argument('--early_stop_patience', default=50, type=float, help='window of epochs to check if the validation loss decreases')
    
    # Architecture parameters
    parser.add_argument('--n_neurons', default=70, type=int, help='number of neurons per hidden layer')
    parser.add_argument('--code_size', default=25, type=int, help='number of neurons in the code layer')

    args = parser.parse_args()

    main(args)

