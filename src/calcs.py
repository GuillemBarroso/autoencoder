import torch
import numpy as np
import copy

def computeLosses(pred, input, model, reg, reg_coef):
        pred = torch.reshape(pred, input.shape)
        loss_image = torch.mean((pred-input)**2)
        if reg:
            # Zaragoza regularisation: 
            # loss_reg = torch.mean(torch.abs(code))

            # L2 regularisation
            loss_reg = sum(p.pow(2.0).sum() for p in model.parameters())

            # L1 regularization
            # loss_reg = sum(p.abs().sum() for p in model.parameters())

            # L1 regularisation but not on activation funct parameters
            # loss_reg = sum(p.abs().sum() for name, p in model.named_parameters() if not 'param_relu' in name or 'param_sigmoid' in name)
            
            # L0 regularisation not working because its non differentiable
            # loss_reg = 0
            # for param in model.parameters():
            #     loss_reg += torch.count_nonzero(param)

            # L0 regularisation from paper
            loss_tot = loss_image + reg_coef*loss_reg
        else:
            loss_tot = loss_image; loss_reg = torch.tensor(0.0)
        return loss_tot, loss_image, loss_reg

def codeInfo(code):
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