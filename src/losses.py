import torch
import numpy as np


def computeLosses(out, input, autoencoder, reg, reg_coef, mode, n_train_params, n_biases, code_coef, bias_ord, bias_coef):
    loss = []
    img_size = np.prod(input.shape[1:])

    # image loss
    X_nn = torch.reshape(out[0], input.shape)
    loss_image_nn = torch.mean((X_nn-input)**2) 
    loss.append(loss_image_nn)

    # combined image loss
    if mode == 'combined':
        X_mu = torch.reshape(out[1], input.shape)
        code_size = out[2].shape[1]
        loss_image_mu = torch.mean((X_mu-input)**2) 
        loss.append(loss_image_mu) 
        loss_code = torch.mean((out[2]-out[3])**2 * (img_size/code_size) * code_coef)  
        loss.append(loss_code)

    # loop over layers and store weights and biases (only if reg or bias_ord activated)
    if reg or bias_ord:
        params = torch.empty(0, dtype=torch.float64)
        biases = torch.empty(0, dtype=torch.float64)
        biases_shift = torch.empty(0, dtype=torch.float64)

        for model in autoencoder:
            if model:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        params = torch.cat((params, torch.flatten(param)), 0)
                    if 'bias' in name:
                        params = torch.cat((params, param), 0)
                        biases = torch.cat((biases, param[:-1]), 0)
                        biases_shift = torch.cat((biases_shift, param[1:]), 0)
    
    # add L1 regularisation term to loss 
    if reg:
        loss_reg_opt = torch.sum(torch.abs(params)) * reg_coef * img_size / n_train_params
        loss.append(loss_reg_opt)
    
    # add bias ordering term to loss through Moreau-Yosida-like regularisation
    if bias_ord:
        loss_bias_opt = biases_shift-biases
        loss_bias_opt[loss_bias_opt>0] = 0
        loss_bias_opt = torch.sum(torch.pow(loss_bias_opt, 2.0)) * bias_coef * img_size / n_biases
        loss.append(loss_bias_opt)

    # Compute total loss to be optimised and place as the 0th entry of loss list
    # loss_tot = torch.sum(torch.tensor(loss, requires_grad=True))
    loss_tot = sum(loss)
    loss.insert(0, loss_tot)

    return loss

def computeErrors(n_test, ref, out):
    rel_e_L1 = 0
    rel_e_L2 = 0
    rel_e_infty = 0
    for i in range(n_test):
        rel_e_L1 += torch.sum(torch.abs(ref[i]-out[i]))/torch.sum(torch.abs(ref[i]))
        rel_e_L2 += torch.sum((ref[i]-out[i])**2)/torch.sum((ref[i])**2)
        rel_e_infty += torch.max(torch.abs(ref[i]-out[i]))/torch.max(torch.abs(ref[i]))
    rel_e_L1 /= n_test
    rel_e_L2 /= n_test
    rel_e_infty /= n_test

    return rel_e_L1, rel_e_L2, rel_e_infty