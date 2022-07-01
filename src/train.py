import torch
import math
from src.postprocess import plotTraining

class Operate(object):
    def __init__(self, model, x_train, x_val, args):
        self.optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self.model = model
        self.x_train = x_train
        self.x_val = x_val
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.reg_coef = args.reg_coef
        self.early_stop_patience = args.early_stop_patience
        self.loss_train = []
        self.loss_val = []
        self.best_loss_val = None
        self.stop_training = None


    def train(self):
        for e in range(self.epochs):
            print(f"Epoch {e+1}\n-------------------------------")
            self.__trainEpoch()
            self.__testEpoch()
            self.__checkEarlyStop(e)
            if self.stop_training:
                break
        print("Training finished!")
        plotTraining(e+1, self.loss_train, self.loss_val)

    def __trainEpoch(self):
        n_batches = self.__getNumBatches()

        for batch in range(n_batches):
            X = self.__getBatchData(batch)

            loss, loss_image, loss_reg = self.__forwardPassAndGetLoss(X)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"tot_loss: {loss:>7f}, image_loss: {loss_image:>7f}, reg_loss: {loss_reg:>7f},  images: [{current:>5d}/{len(self.x_train):>5d}]")
        self.loss_train.append(loss)

    def __testEpoch(self):
        with torch.no_grad():
            loss, loss_image, loss_reg = self.__forwardPassAndGetLoss(self.x_val.data)
            self.loss_val.append(loss)
            loss_val_image = loss_image
            loss_val_reg = loss_reg

        print(f"Test Error: \ntot_loss: {self.loss_val[-1]:>8f}, image_loss: {loss_val_image:>8f}, reg_loss: {loss_val_reg:>8f} \n")

    def __getNumBatches(self):
        return math.ceil(len(self.x_train)/self.batch_size)
    
    def __getBatchData(self, batch):
        ini = batch*self.batch_size
        end = ini+self.batch_size
        if end > len(self.x_train): end = len(self.x_train)
        return self.x_train[ini:end]

    def __forwardPassAndGetLoss(self, X):
        pred, code = self.model(X)
        loss_image, loss_reg = self.__computeLosses(pred, X, code)
        return loss_image + self.reg_coef*loss_reg, loss_image, loss_reg

    def __computeLosses(self, pred, input, code):
        pred = torch.reshape(pred, input.shape)
        loss_image = torch.mean((pred-input)**2)
        loss_reg = torch.mean(torch.abs(code))
        return loss_image, loss_reg

    def __checkEarlyStop(self, e):
        if e >= self.early_stop_patience:
            self.best_loss_val = min(self.loss_val)
            loss_val_window = self.loss_val[-self.early_stop_patience:]
            if min(loss_val_window) > self.best_loss_val: 
                self.stop_training = True
                print('Early stop triggered')


