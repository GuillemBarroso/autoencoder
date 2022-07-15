import matplotlib.pyplot as plt
import torch
import pandas as pd
import dataframe_image as dfi


def reshape(x, size):
    return torch.reshape(x, size)

def summaryInfo(data, name, verbose):
    df = pd.DataFrame(data, columns=['Parameter', 'Value'])
    if verbose:
        print(df)
    dfi.export(df, name)

def plotImage(data, nRows, n_disp, count):
        ax = plt.subplot(nRows, n_disp, count)
        plt.imshow(data)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

def plotZeroCode(code_size, plot_code, colour):
    count = 0
    for y in range(code_size):
        for x in range(code_size):  
            if count in plot_code:
                plt.scatter(x,y, color=colour,s=5)
            count += 1

def plotting(input, code_nn, code_mu, X_nn, X_mu, img_names, zero_code, zero_code_mu, data_class):
    plotNames = ['X', 'code_nn', 'X_NN', 'code_mu', 'X_mu', 'fig data']
    nRows = len(plotNames)
    n_disp = len(input)
    plt.figure(figsize=(20, 8))
    for i in range(n_disp):
        plotImage(input[i], nRows, n_disp, i+1)
        plotImage(code_nn[i], nRows, n_disp, i+1+n_disp)
        plotZeroCode(len(code_nn.data[0]), zero_code, 'blue')
        plotImage(X_nn[i], nRows, n_disp, i+1+2*n_disp)
        plotImage(code_mu[i], nRows, n_disp, i+1+3*n_disp)
        plotZeroCode(len(code_mu.data[0]), zero_code_mu, 'blue')
        plotImage(X_mu[i], nRows, n_disp, i+1+4*n_disp)

        # Display error for each test image
        mus = data_class.getMusFromImgName(img_names[i])
        img_error = [torch.mean((X_nn[i]-input[i,:,:,0])**2), torch.mean((X_mu[i]-input[i,:,:,0])**2)]

        imageError_nn = '{:.2}'.format(img_error[0].item())
        imageError_mu = '{:.2}'.format(img_error[1].item())
        ax = plt.subplot(nRows, n_disp, i + 1 + 5*n_disp)
        for imu in range(len(mus)):
            ax.text(0.15,0.5-(imu*0.13),'mu{} = {}'.format(imu+1, mus[imu]))
        ax.text(0.15,0.1,'loss_image_nn = {}'.format(imageError_nn))
        ax.text(0.15,-0.08,'loss_image_mu = {}'.format(imageError_mu))
        ax.axis('off')
    addPlotNames(plotNames)
    savePlot('predictsPlot.png')

def addPlotNames(plotNames):
    for i, plotName in enumerate(reversed(plotNames)):
        plt.text(0.1, 0.13+0.14*i, plotName, fontsize=12, transform=plt.gcf().transFigure, rotation=90)

def plotTraining(epochs, hist):
    x_train = range(epochs)
    x_val = range(1,epochs+1)
    
    plt.figure()
    for loss in hist.loss_train:
        plt.plot(x_train, loss)

    for loss in hist.loss_val:
        plt.plot(x_val, loss)
    plt.legend([x + ' train' for x in hist.encoder.loss_names] + [x + ' val' for x in hist.encoder.loss_names], loc='upper right')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('log(loss)')
    plt.title('Training and validation losses')

    # lims = ax.get_ylim()
    # if lims[1] > 0.5:
    #     limsPlot = [lims[0], 0.5]
    # else:
    #     limsPlot = lims
    # ax.set_ylim(limsPlot)
    savePlot('trainPlot.png')

    if hist.encoder.param_activation:
        plt.figure()
        plt.plot(range(epochs), hist.alphas[0])
        plt.plot(range(epochs), hist.alphas[1])
        plt.title('Activation function parameters')
        plt.ylabel('alpha')
        plt.xlabel('epoch')
        plt.legend(['alpha ReLu', 'alpha Sigmoid'], loc='upper right')
        savePlot('alphasPlot.png')

def savePlot(name):
    plt.savefig('results/{}'.format(name))

def plotShow():
    plt.show()

def addLossesToList(losses, mode, loss_names, data=[]):
    for i, loss in enumerate(losses): 
        data.append(['{} {}'.format(loss_names[i], mode), '{:.2}'.format(loss[-1])])
    return data

def storeLossInfo(losses, lossStore):
        for i, loss in enumerate(losses):
            lossStore[i].append(loss.item())