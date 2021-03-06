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

def codeInfo(code):
            code = np.reshape(code, [code.shape[0], code.shape[1]* code.shape[2]])
            avg = np.true_divide(code.sum(0), code.shape[0])
            active_code_flag = np.nonzero(avg)
            zero_code_flag = np.where(avg==0)[0]
            code_size = len(active_code_flag)
            avg_code_mag = np.true_divide(abs(avg).sum(),(avg!=0).sum())
            return zero_code_flag, code_size, avg_code_mag

def truncCode(code, code_size, active_code_size, threshold=0.1):
    # Detect dimensionality
    latent = np.sum(code.numpy()**2, axis=0)**0.5
    rel_latent = latent/np.max(latent)

    trunc_code_flag = []
    trunc_code = copy.deepcopy(code)

    # Truncate code
    for i in range(code_size):
        if 0.0 < rel_latent[i] <= threshold: 
            trunc_code_flag.append(i)
            for k in range(code.shape[0]):
                trunc_code[k][i] = 0
            
    latent_trunc_size = active_code_size - len(trunc_code_flag)
    return trunc_code_flag, trunc_code, latent_trunc_size