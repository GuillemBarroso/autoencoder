import torch

def computeLosses(out, input, autoencoder, reg, reg_coef, mode, n_train_params, n_biases, code_coef, bias_ord, bias_coef):
    loss = []
    img_size = input.shape[1]*input.shape[2]
    X_nn = torch.reshape(out[0], input.shape)
    loss_image_nn = torch.mean((X_nn-input)**2) 
    loss.append(loss_image_nn)

    if mode == 'combined':
        X_mu = torch.reshape(out[1], input.shape)
        code_size = out[2].shape[1]
        loss_image_mu = torch.mean((X_mu-input)**2) 
        loss.append(loss_image_mu) 
        loss_code = torch.mean((out[2]-out[3])**2 * (img_size/code_size) * code_coef)  
        loss.append(loss_code)

    # Add regularisation term
    if reg:
        # L2 regularisation
        # loss_reg = 0
        # for model in autoencoder:
        #     loss_reg += sum(p.pow(2.0).sum()* reg_coef / n_train_params for p in model.parameters())

        # L1 regularization
        loss_reg = torch.tensor([0.0], requires_grad=True)
        for model in autoencoder:
            if model:
                loss_reg = loss_reg + sum(p.abs().sum() for p in model.parameters()) * reg_coef * (img_size / n_train_params)
        
        loss.append(loss_reg)

    # Bias ordering
    if bias_ord:
        loss_bias = torch.tensor([0.0], requires_grad=True)
        for model in autoencoder:
            if model:
                for name, param in model.named_parameters():
                    if 'bias' in name:
                        for i in range(len(param)-1):
                            loss_bias = loss_bias + torch.pow(torch.min(torch.FloatTensor([param[i+1]-param[i], 0])), 2) * bias_coef * (img_size / n_biases)
        
        loss.append(loss_bias)

    
    # Compute total loss to be optimised and place as the 0th entry of loss list
    loss_tot = sum(loss)
    loss.insert(0, loss_tot)

    return loss