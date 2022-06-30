import torch
import math


class Operate(object):
    def __init__(self, model, x_train, x_test, args):
        self.optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.reg_coef = args.reg_coef

    def train(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.trainEpoch()
            self.testEpoch()
        print("Training finished!")

    def trainEpoch(self):
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

    def testEpoch(self):
        with torch.no_grad():
            loss, loss_image, loss_reg = self.__forwardPassAndGetLoss(self.x_test.data)
            test_loss = loss
            test_loss_image = loss_image
            test_loss_reg = loss_reg

        print(f"Test Error: \ntot_loss: {test_loss:>8f}, image_loss: {test_loss_image:>8f}, reg_loss: {test_loss_reg:>8f} \n")

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

    def __checkEarlyStop():
        pass


