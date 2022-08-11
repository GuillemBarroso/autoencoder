import os
import timeit
import torch
from torch.utils.data import DataLoader

from src.postprocess import plotTraining, summaryInfo, addLossesToList, storeLossInfo, reshape
from src.losses import computeLosses


class Train(object):
    def __init__(self, autoencoder, data, args):
        self.x_train = data.x_train
        self.x_val = data.x_val
        self.mus_train = data.mus_train
        self.mus_val = data.mus_val
        self.fig_path = data.fig_path
        self.name = data.name
        self.fig_path = data.fig_path
        self.dataset = args.dataset
        self.random_test_data = args.random_test_data
        self.random_seed = args.random_seed
        self.manual_data = args.manual_data
        self.layers = args.layers
        self.layers_mu = args.layers_mu
        self.learning_rate = args.learning_rate
        self.lr_epoch_milestone = args.lr_epoch_milestone
        self.lr_red_coef = args.lr_red_coef
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.reg_coef = args.reg_coef
        self.reg = args.reg
        self.bias_coef = args.bias_coef
        self.bias_ord = args.bias_ord
        self.early_stop_patience = args.early_stop_patience
        self.early_stop_tol = args.early_stop_tol
        self.verbose = args.verbose
        self.plot = args.plot
        self.act_hid = args.act_hid
        self.act_out = args.act_out
        self.act_code = args.act_code
        self.mode = args.mode
        self.code_size = args.layers[-1]
        self.code_coef = args.code_coef
        self.save_model = args.save_model
        self.save_fig = args.save_fig
        self.alphas = [[], []]
        self.best_loss_val = None
        self.stop_training = None
        self.train_time = None
        self.early_stop_count = 0
        self.loss_prev_best = self.early_stop_tol*1e15
        self.n_train_params = autoencoder.n_train_params
        self.n_biases = autoencoder.n_biases
        self.model_path = f'{args.model_dir}/{args.dataset}'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Load arch information depending on the autoencoder's mode
        self.autoencoder = autoencoder
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.parameter = autoencoder.parameter

        self.optim_decoder, self.scheduler = self.__initialiseModel(self.decoder)

        self.idx_early_stop = self.autoencoder.idx_early_stop
        self.loss_train = [[] for x in range(len(self.autoencoder.loss_names))]
        self.loss_val = [[] for x in range(len(self.autoencoder.loss_names))]

        if self.mode == 'standard' or self.mode == 'combined':
            self.optim_encoder, self.scheduler = self.__initialiseModel(self.encoder)

        if self.mode == 'combined' or self.mode == 'parametric':
            self.optim_param, self.scheduler = self.__initialiseModel(self.parameter)

        # Use DataLoaders for batch training
        self.x_loader = DataLoader(self.x_train, batch_size=self.batch_size, shuffle=False)
        self.mus_loader = DataLoader(self.mus_train, batch_size=self.batch_size, shuffle=False)
        
        # Training
        try:
            if self.mode == 'standard':
                autoencoder.encoder.load_state_dict(torch.load(f"{self.model_path}/encoder_{self.name}"))
                autoencoder.decoder.load_state_dict(torch.load(f"{self.model_path}/decoder_{self.name}"))
            elif self.mode == 'combined':
                autoencoder.encoder.load_state_dict(torch.load(f"{self.model_path}/encoder_{self.name}"))
                autoencoder.decoder.load_state_dict(torch.load(f"{self.model_path}/decoder_{self.name}"))
                autoencoder.parameter.load_state_dict(torch.load(f"{self.model_path}/parameter_{self.name}"))
            elif self.mode == 'parametric':
                autoencoder.decoder.load_state_dict(torch.load(f"{self.model_path}/decoder_{self.name}"))
                autoencoder.parameter.load_state_dict(torch.load(f"{self.model_path}/parameter_{self.name}"))
            print("Existing model loaded. Training skipped.")

        except FileNotFoundError:
            print('Model not found. Starting training.')
            self.train()

            # Save model
            if self.save_model:
                if self.mode == 'standard':
                    torch.save(self.autoencoder.encoder.state_dict(), f"{self.model_path}/encoder_{self.name}")
                    torch.save(self.autoencoder.decoder.state_dict(), f"{self.model_path}/decoder_{self.name}")
                elif self.mode == 'combined':
                    torch.save(self.autoencoder.encoder.state_dict(), f"{self.model_path}/encoder_{self.name}")
                    torch.save(self.autoencoder.decoder.state_dict(), f"{self.model_path}/decoder_{self.name}")
                    torch.save(self.autoencoder.parameter.state_dict(), f"{self.model_path}/parameter_{self.name}")
                elif self.mode == 'parametric':
                    torch.save(self.autoencoder.decoder.state_dict(), f"{self.model_path}/decoder_{self.name}")
                    torch.save(self.autoencoder.parameter.state_dict(), f"{self.model_path}/parameter_{self.name}")
                elif self.mode == 'staggered':
                    torch.save(self.autoencoder.parameter.state_dict(), f"{self.model_path}/parameter_{self.name}")
                print("Existing model loaded. Training skipped.")

    def train(self):
        def __summary():
            name = f'{self.fig_path}/trainTable_{self.name}.png'
            data = [['epochs', self.epochs],
                ['batch size', self.batch_size],
                ['early stop patience', '{} epochs'.format(self.early_stop_patience)],
                ['early stop tol', '{:.0e} %'.format(self.early_stop_tol)],
                ['initial learning rate', '{:.0e}'.format(self.learning_rate)],
                ['epochs lr reduction', '{}'.format(self.lr_epoch_milestone)],
                ['lr reduction factor', '{:.0e}'.format(self.lr_red_coef)],
                ['training time', '{:.2}s/{:.3}min'.format(self.train_time, self.train_time/60)],
                ['regularisation', '{}'.format(self.reg)],
                ['bias ordering', '{}'.format(self.bias_ord)],
            ]
            if self.reg:
                data.append(['regularisation coef', '{:.0e}'.format(self.reg_coef)])
            if self.mode == 'combined':
                data.append(['code coef', self.code_coef])
            if self.bias_ord:
                data.append(['bias coef', self.bias_coef])

            if self.act_hid == 'param_relu' or self.act_code == 'param_relu':
                data.append(['relu optim alpha', self.autoencoder.param_relu.alpha.item()])
            if self.act_out == 'param_sigmoid':
                data.append(['sigmoid optim alpha', self.autoencoder.param_sigmoid.alpha.item()])

            data = addLossesToList(self.loss_train, 'train', self.autoencoder.loss_names, data)
            data = addLossesToList(self.loss_val, 'val', self.autoencoder.loss_names, data)
            summaryInfo(data, name, self.verbose, self.save_fig)

        start = timeit.default_timer()
        for e in range(self.epochs):
            if self.verbose:
                print(f"Epoch {e+1}\n-------------------------------")
            self.__trainEpoch()
            self.__valEpoch()
            self.__checkEarlyStop()
            if self.stop_training:
                break
        self.train_time = timeit.default_timer() - start
        if self.plot:
            plotTraining(e+1, self)
        __summary()

    def __trainEpoch(self):
        for X, mus in zip(self.x_loader, self.mus_loader):

            loss = self.__evaluate(X, mus)
            
            # Set grads to zero
            self.optim_decoder.zero_grad()
            if self.mode == 'standard' or self.mode == 'combined':
                self.optim_encoder.zero_grad()
            if self.mode == 'combined' or self.mode == 'parametric':
                self.optim_param.zero_grad()

            # Backward for the total loss (first entry in loss list)
            loss[0].backward()

            # Update models' weights
            self.optim_decoder.step()
            if self.mode == 'standard' or self.mode == 'combined':
                self.optim_encoder.step()
            if self.mode == 'combined' or self.mode == 'parametric':
                self.optim_param.step()
            self.scheduler.step()

            if self.verbose:
                self.__printTrainInfo(loss)

        storeLossInfo(loss, self.loss_train)

        if self.autoencoder.param_activation:
            self.alphas[0].append(self.autoencoder.param_relu.alpha.item())
            self.alphas[1].append(self.autoencoder.param_sigmoid.alpha.item())

    def __valEpoch(self):
        with torch.no_grad():
            loss = self.__evaluate(self.x_val.data, self.mus_val)
        storeLossInfo(loss, self.loss_val)

        if self.verbose:
                self.__printTrainInfo(loss, val=True)

    def __evaluate(self, X, mus):
        if self.mode == 'combined':
            code_nn = self.encoder(X)
            code_mu = self.parameter(mus)
            out_nn = reshape(self.decoder(code_nn), X.shape)
            out_mu = reshape(self.decoder(code_mu), X.shape)
            out = [out_nn, out_mu, code_nn, code_mu]
        elif self.mode == 'standard':
            code = self.encoder(X)
            out = [self.decoder(code)]
        elif self.mode == 'parametric':
            code = self.parameter(mus)
            out = [self.decoder(code)]
        else: 
            raise NotImplementedError

        loss = computeLosses(out, X, self.autoencoder.models, self.reg, self.reg_coef, self.mode, self.n_train_params, self.n_biases, self.code_coef, self.bias_ord, self.bias_coef)
        return loss

    def __checkEarlyStop(self):
        loss_current = self.loss_val[self.idx_early_stop][-1]
        
        if (self.loss_prev_best - loss_current)/self.loss_prev_best < self.early_stop_tol/100:
            self.early_stop_count += 1 
        else:
            self.early_stop_count = 0
            self.loss_prev_best = loss_current

        if self.early_stop_count == self.early_stop_patience:
            self.stop_training = True
            if self.verbose:
                print('Early stop triggered')

    def __printTrainInfo(self, losses, val=False):
        info = f"ValError:\n" if val else f""

        for i, loss in enumerate(losses):
            info += "{}: {:.6}, ".format(self.autoencoder.loss_names[i], loss.item())
        print(info[:-2])

    def __initialiseModel(self, model):
        # optim = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=0)
        optim = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.lr_epoch_milestone, gamma=self.lr_red_coef)
        return optim, scheduler