import torch
import math
import timeit
from src.postprocess_mu import plotTraining, summaryInfo, addLossesToList, storeLossInfo
from src.calcs_mu import computeLosses


class Model(object):
    def __init__(self, model, data, args):
        self.learning_rate = args.learning_rate
        self.lr_epoch_milestone = args.lr_epoch_milestone
        self.lr_red_coef = args.lr_red_coef
        self.model = model
        self.x_train = data.x_train
        self.x_val = data.x_val
        self.mus_train = data.mus_train
        self.mus_val = data.mus_val
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
        self.alphas = [[], []]
        self.best_loss_val = None
        self.stop_training = None
        self.train_time = None
        self.early_stop_count = 0
        self.loss_prev_best = self.early_stop_tol*1e15
        
        self.loss_train = [[] for x in range(len(self.model.loss_names))]
        self.loss_val = [[] for x in range(len(self.model.loss_names))]

        self.optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=self.lr_epoch_milestone, gamma=self.lr_red_coef)

    def train(self):
        def __summary():
            name = 'results/trainTable.png'
            data = [['epochs', self.epochs],
            ['batch size', self.batch_size],
            ['early stop patience', '{} epochs'.format(self.early_stop_patience)],
            ['early stop tol', '{:.0e}'.format(self.early_stop_tol)],
            ['initial learning rate', '{:.0e}'.format(self.learning_rate)],
            ['epochs lr reduction', '{}'.format(self.lr_epoch_milestone)],
            ['lr reduction factor', '{:.0e}'.format(self.lr_red_coef)],
            ['training time', '{:.2}s/{:.3}min'.format(self.train_time, self.train_time/60)],
            ['regularisation', '{}'.format(self.reg)],
            ]
            if self.reg:
                data.append(['regularisation coef', '{:.0e}'.format(self.reg_coef)])

            if self.act_hid == 'param_relu' or self.act_code == 'param_relu':
                data.append(['relu optim alpha', self.model.param_relu.alpha.item()])
            if self.act_out == 'param_sigmoid':
                data.append(['sigmoid optim alpha', self.model.param_sigmoid.alpha.item()])
            data = addLossesToList(self.loss_train, 'test', self.model.loss_names, data)
            data = addLossesToList(self.loss_val, 'val', self.model.loss_names, data)
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

            self.optimiser.zero_grad()
            loss[0].backward()
            self.optimiser.step()
            self.scheduler.step()
            if self.verbose:
                self.__printTrainInfo(loss)

        storeLossInfo(loss, self.loss_train)

        if self.model.param_activation:
            self.alphas[0].append(self.model.param_relu.alpha.item())
            self.alphas[1].append(self.model.param_sigmoid.alpha.item())

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
        out = self.model(X, mus)
        return computeLosses(out, X, self.model, self.reg, self.reg_coef)

    def __checkEarlyStop(self):
        loss_current = self.loss_val[2][-1]
        
        if self.loss_prev_best - loss_current < self.early_stop_tol:
            self.early_stop_count += 1 
        else:
            self.early_stop_count = 0
            self.loss_prev_best = loss_current

        if self.early_stop_count == self.early_stop_patience:
            self.stop_training = True
            if self.verbose:
                print('Early stop triggered')

    def __printTrainInfo(self, loss, val=False):
        info = f"ValError:\n" if val else f""

        for i, loss in enumerate(loss):
            info += "{}: {:.6}, ".format(self.model.loss_names[i], loss)
        print(info[:-2])