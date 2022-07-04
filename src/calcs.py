import torch
import numpy as np
import copy

def computeLosses(pred, input, code, reg_coef):
        pred = torch.reshape(pred, input.shape)
        loss_image = torch.mean((pred-input)**2)
        loss_reg = torch.mean(torch.abs(code))
        loss_tot = loss_image + reg_coef*loss_reg
        return loss_tot, loss_image, loss_reg

def codeInfo(code):
            avg = np.true_divide(code.sum(0), code.shape[0])
            active_code_flag = np.nonzero(avg)
            code_size = len(active_code_flag)
            avg_code_mag = np.true_divide(abs(avg).sum(),(avg!=0).sum())
            return active_code_flag, code_size, avg_code_mag

def truncCode(code, code_size, threshold=0.1):
    # Detect dimensionality
    latent = np.sum(code.numpy()**2, axis=0)**0.5
    rel_latent = latent/np.max(latent)

    trunc_code_flag = []
    trunc_code = copy.deepcopy(code)

    # Truncate code
    for i in range(code_size):
        if rel_latent[i] <= threshold: 
            for k in range(code.shape[0]):
                trunc_code[k][i] = 0
        else:
            trunc_code_flag.append(i)
            
    latent_trunc_size = len(trunc_code_flag)
    return trunc_code_flag, trunc_code, latent_trunc_size