import torch
import math
import timeit
from src.postprocess import plotTraining, summaryInfo, addLossesToList, storeLossInfo, reshape
from src.losses import computeLosses


class Model(object):
    def __init__(self, autoencoder, data, args):
        self.x_train = data.x_train
        self.x_val = data.x_val
        self.mus_train = data.mus_train
        self.mus_val = data.mus_val
        self.learning_rate = args.learning_rate
        self.lr_epoch_milestone = args.lr_epoch_milestone
        self.lr_red_coef = args.lr_red_coef
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.reg_coef = args.reg_coef
        self.reg = args.reg
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
        self.alphas = [[], []]
        self.best_loss_val = None
        self.stop_training = None
        self.train_time = None
        self.early_stop_count = 0
        self.loss_prev_best = self.early_stop_tol*1e15
        self.n_train_params = 0

        self.autoencoder = autoencoder
        self.encoder = autoencoder[0]
        self.decoder = autoencoder[1]           
        self.idx_early_stop = self.encoder.idx_early_stop
        self.loss_train = [[] for x in range(len(self.encoder.loss_names))]
        self.loss_val = [[] for x in range(len(self.encoder.loss_names))]

        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate, weight_decay=0)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim_encoder, milestones=self.lr_epoch_milestone, gamma=self.lr_red_coef)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim_decoder, milestones=self.lr_epoch_milestone, gamma=self.lr_red_coef)
        if self.mode == 'combined':
            self.parameter = autoencoder[2]
            self.n_train_params += self.__count_parameters(self.parameter)
            self.optim_param = torch.optim.Adam(self.parameter.parameters(), lr=self.learning_rate, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim_param, milestones=self.lr_epoch_milestone, gamma=self.lr_red_coef)

        self.n_train_params += self.__count_parameters(self.encoder)
        self.n_train_params += self.__count_parameters(self.decoder)
        data.n_train_params = self.n_train_params # store in data to be used during predicitons

    def train(self):
        def __summary():
            name = 'results/trainTable.png'
            data = [['epochs', self.epochs],
                ['batch size', self.batch_size],
                ['num train params', self.n_train_params],
                ['early stop patience', '{} epochs'.format(self.early_stop_patience)],
                ['early stop tol', '{:.0e} %'.format(self.early_stop_tol)],
                ['initial learning rate', '{:.0e}'.format(self.learning_rate)],
                ['epochs lr reduction', '{}'.format(self.lr_epoch_milestone)],
                ['lr reduction factor', '{:.0e}'.format(self.lr_red_coef)],
                ['training time', '{:.2}s/{:.3}min'.format(self.train_time, self.train_time/60)],
                ['regularisation', '{}'.format(self.reg)],
            ]
            if self.reg:
                data.append(['regularisation coef', '{:.0e}'.format(self.reg_coef)])
            if self.mode == 'combined':
                data.append(['code coef', self.code_coef])

            if self.act_hid == 'param_relu' or self.act_code == 'param_relu':
                data.append(['relu optim alpha', self.encoder.param_relu.alpha.item()])
            if self.act_out == 'param_sigmoid':
                data.append(['sigmoid optim alpha', self.encoder.param_sigmoid.alpha.item()])
            data = addLossesToList(self.loss_train, 'train', self.encoder.loss_names, data)
            data = addLossesToList(self.loss_val, 'val', self.encoder.loss_names, data)
            summaryInfo(data, name, self.verbose)

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
        n_batches = self.__getNumBatches()

        for batch in range(n_batches):
            X = self.__getBatchData(self.x_train, batch)
            mus = self.__getBatchData(self.mus_train, batch)

            loss = self.__evaluate(X, mus)

            self.optim_encoder.zero_grad()
            self.optim_decoder.zero_grad()
            if self.mode == 'combined':
                self.optim_param.zero_grad()
            loss[0].backward()
            self.optim_encoder.step()
            self.optim_decoder.step()
            if self.mode == 'combined':
                self.optim_param.step()
            self.scheduler.step()

            if self.verbose:
                self.__printTrainInfo(loss)

        storeLossInfo(loss, self.loss_train)

        if self.encoder.param_activation:
            self.alphas[0].append(self.encoder.param_relu.alpha.item())
            self.alphas[1].append(self.encoder.param_sigmoid.alpha.item())

    def __valEpoch(self):
        with torch.no_grad():
            loss = self.__evaluate(self.x_val.data, self.mus_val)
        storeLossInfo(loss, self.loss_val)

        if self.verbose:
                self.__printTrainInfo(loss, val=True)

    def __getNumBatches(self):
        return math.ceil(len(self.x_train)/self.batch_size)
    
    def __getBatchData(self, x, batch):
        ini = batch*self.batch_size
        end = ini+self.batch_size
        if end > len(x): end = len(x)
        return x[ini:end]

    def __evaluate(self, X, mus):
        if self.mode == 'combined':
            code_nn = self.encoder(X)
            code_mu = self.parameter(mus)
            out_nn = reshape(self.decoder(code_nn), X.shape)
            out_mu = reshape(self.decoder(code_mu), X.shape)
            out = [out_nn, out_mu, code_nn, code_mu]
        elif self.mode == 'standard':
            code = self.encoder(X)
            out = [self.decoder(code), code]
        else: 
            raise NotImplementedError

        loss = computeLosses(out, X, self.autoencoder, self.reg, self.reg_coef, self.mode, self.n_train_params, self.code_coef)
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
            info += "{}: {:.6}, ".format(self.encoder.loss_names[i], loss)
        print(info[:-2])

    def __count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad)