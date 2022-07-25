import torch
import numpy as np
import copy

def computeLosses(out, input, autoencoder, reg, reg_coef, mode, n_train_params, weights):
    loss = []
    img_size = input.shape[1]*input.shape[2]
    X_nn = torch.reshape(out[0], input.shape)
    loss_image_nn = torch.mean( (X_nn-input)**2 / img_size * weights[0]) 
    loss.append(loss_image_nn)

    if mode == 'parametric':
        X_mu = torch.reshape(out[1], input.shape)
        loss_image_mu = torch.mean((X_mu-input)**2 / img_size * weights[1])  
        # loss_image_mu = torch.mean((X_mu-input).pow(2.0)) / img_size * weights[1]
        loss.append(loss_image_mu) 
        loss_code = torch.mean((out[2]-out[3])**2 / out[2].shape[1] * weights[2])  
        # loss_code = torch.mean((out[2]-out[3])**2) * weights[2]
        loss.append(loss_code)

    # Add regularisation term
    if reg:
        # L2 regularisation
        # loss_reg = 0
        # for model in autoencoder:
        #     loss_reg += sum(p.pow(2.0).sum()* reg_coef / n_train_params for p in model.parameters())

        # L1 regularization
        loss_reg = 0
        for model in autoencoder:
            loss_reg += sum(p.abs().sum()* reg_coef / n_train_params for p in model.parameters())
        
        loss.append(loss_reg)
    
    # Compute total loss to be optimised and place as the 0th entry of loss list
    loss_tot = sum(loss)
    loss.insert(0, loss_tot)

    return loss