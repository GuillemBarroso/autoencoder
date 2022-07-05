import torch
import math
import timeit
from src.postprocess import plotTraining, summaryInfo
from src.calcs import computeLosses


class Model(object):
    def __init__(self, model, x_train, x_val, args):
        self.learning_rate = args.learning_rate
        self.epoch_milestone = args.epoch_milestone
        self.lr_red_coef = args.lr_red_coef
        self.model = model
        self.x_train = x_train
        self.x_val = x_val
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.reg_coef = args.reg_coef
        self.early_stop_patience = args.early_stop_patience
        self.early_stop_tol = args.early_stop_tol
        self.verbose = args.verbose
        self.plot = args.plot
        self.loss_train = []
        self.loss_train_image = []
        self.loss_train_reg = []
        self.loss_val = []
        self.loss_val_image = []
        self.loss_val_reg = []
        self.best_loss_val = None
        self.stop_training = None
        self.train_time = None
        self.early_stop_count = 0
        self.loss_prev_best = self.early_stop_tol*1e15
        self.optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=self.epoch_milestone, gamma=self.lr_red_coef)

    def train(self):
        def __summary():
            name = 'results/trainTable.png'
            data = [['epochs', self.epochs],
            ['batch size', self.batch_size],
            ['early stop patience', '{} epochs'.format(self.early_stop_patience)],
            ['early stop tol', '{:.0e}'.format(self.early_stop_tol)],
            ['initial learning rate', '{:.0e}'.format(self.learning_rate)],
            ['epochs lr reduction', '{}'.format(self.epoch_milestone)],
            ['lr reduction factor', '{:.0e}'.format(self.lr_red_coef)],
            ['training time', '{:.2}s/{:.1}min'.format(self.train_time, self.train_time/60)],
            ['regularisation coef', '{:.0e}'.format(self.reg_coef)],
            ['total train loss', '{:.2}'.format(self.loss_train[-1])],
            ['image train loss', '{:.2}'.format(self.loss_train_image[-1])],
            ['reg train loss', '{:.2}'.format(self.loss_train_reg[-1])],
            ['total val loss', '{:.2}'.format(self.loss_val[-1])],
            ['image val loss', '{:.2}'.format(self.loss_val_image[-1])],
            ['reg val loss', '{:.2}'.format(self.loss_val_reg[-1])],
            ]
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
            plotTraining(e+1, self.loss_train, self.loss_val)
        __summary()

    def __trainEpoch(self):
        n_batches = self.__getNumBatches()

        for batch in range(n_batches):
            X = self.__getBatchData(batch)

            loss, loss_image, loss_reg = self.__evaluate(X)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.scheduler.step()

            if batch % 100 == 0 and self.verbose:
                print(f"tot_loss: {loss.item():>7f}, image_loss: {loss_image.item():>7f}, reg_loss: {loss_reg.item():>7f},  images: [{batch*len(X):>5d}/{len(self.x_train):>5d}]")
        self.loss_train.append(loss.item())
        self.loss_train_image.append(loss_image.item())
        self.loss_train_reg.append(loss_reg.item())

    def __valEpoch(self):
        with torch.no_grad():
            loss, loss_image, loss_reg = self.__evaluate(self.x_val.data)
        self.loss_val.append(loss)
        self.loss_val_image.append(loss_image)
        self.loss_val_reg.append(loss_reg)

        if self.verbose:
            print(f"Val error: \ntot_loss: {loss:>8f}, image_loss: {loss_image:>8f}, reg_loss: {loss_reg:>8f} \n")

    def __getNumBatches(self):
        return math.ceil(len(self.x_train)/self.batch_size)
    
    def __getBatchData(self, batch):
        ini = batch*self.batch_size
        end = ini+self.batch_size
        if end > len(self.x_train): end = len(self.x_train)
        return self.x_train[ini:end]

    def __evaluate(self, X):
        pred, code = self.model(X)
        loss_tot, loss_image, loss_reg = computeLosses(pred, X, code, self.reg_coef)
        return loss_tot, loss_image, loss_reg

    def __checkEarlyStop(self):
        loss_current = self.loss_val[-1]
        
        if self.loss_prev_best - loss_current < self.early_stop_tol:
            self.early_stop_count += 1 
        else:
            self.early_stop_count = 0
            self.loss_prev_best = loss_current

        if self.early_stop_count == self.early_stop_patience:
            self.stop_training = True
            if self.verbose:
                print('Early stop triggered')


