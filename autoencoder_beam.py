import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from src.load_data import Data, DataTorch
from src.model import Autoencoder
from src.train import Operate


def main(args):
    data = Data('beam_homog')
    x_train = DataTorch(data.x_train)
    x_val = DataTorch(data.x_val)
    x_test = DataTorch(data.x_test)

    model = Autoencoder(data, args)
    summary(model, data.resolution)

    Operate(model, x_train, args).train()

    # Make predictions
    n_disp = 6
    val_data = x_val[:n_disp,:,:]

    with torch.no_grad():
        pred, code = model(val_data)

    # Detect dimensionality
    latent = np.sum(code.numpy()**2, axis=0)**0.5
    rel_latent = latent/np.max(latent)

    active_code = np.zeros(args.code_size, dtype=int)

    for i in range(args.code_size):
        if rel_latent[i] > 0.1: active_code[i] = 1

    latent_trunc_dim = sum(active_code)

    code = torch.reshape(code, (code.shape[0], int(np.sqrt(args.code_size)), int(np.sqrt(args.code_size))))
    pred = torch.reshape(pred, (pred.shape[0], data.resolution[0], data.resolution[1]))

    def plotImage(data, nRows, numDisplay, count):
        ax = plt.subplot(nRows, numDisplay, count)
        plt.imshow(data)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    nRows = 6
    plt.figure(figsize=(20, 5))
    for i in range(n_disp):
        plotImage(data.x_val[i], nRows, n_disp, i+1)
        plotImage(code[i], nRows, n_disp, i+1+n_disp)
        plotImage(pred[i], nRows, n_disp, i+1+2*n_disp)
    plt.show()
    test = 1

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Autoencoder for image compression')

    # Training parameters
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--reg_coef', default=1e-4, type=float, help='regularisation coefficient in the code layer')
    parser.add_argument('--batch_size', default=200, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='training learning rate ')
    
    # Architecture parameters
    parser.add_argument('--n_neurons', default=400, type=int, help='number of neurons per hidden layer')
    parser.add_argument('--code_size', default=25, type=int, help='number of neurons in the code layer')

    args = parser.parse_args()

    main(args)

