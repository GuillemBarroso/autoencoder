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

def plotting(input, code, pred, img_names, zero_code, data_class):
    nRows = 4
    plotNames = ['X', 'code', 'X_NN', 'fig data']
    n_disp = len(input)
    plt.figure(figsize=(20, 8))
    for i in range(n_disp):
        plotImage(input[i], nRows, n_disp, i+1)
        plotImage(code[i], nRows, n_disp, i+1+n_disp)
        plotZeroCode(len(code.data[0]), zero_code, 'blue')
        plotImage(pred[i], nRows, n_disp, i+1+2*n_disp)

        # Display error for each test image
        mus = data_class.getMusFromImgName(img_names[i])
        img_error = torch.mean((pred[i]-input[i,:,:,0])**2)

        imageError = '{:.2}'.format(img_error.item())
        ax = plt.subplot(nRows, n_disp, i + 1 + 3*n_disp)
        for imu in range(len(mus)):
            ax.text(0.15,0.5-(imu*0.2),'mu{} = {}'.format(imu+1, mus[imu]))
        ax.text(0.15,0.1,'loss = {}'.format(imageError))
        ax.axis('off')
    addPlotNames(plotNames)
    savePlot('predictsPlot.png')

def addPlotNames(plotNames):
    for i, plotName in enumerate(reversed(plotNames)):
        plt.text(0.1, 0.12+0.23*i, plotName, fontsize=12, transform=plt.gcf().transFigure, rotation=90)

def plotTraining(epochs, hist):
    x_train = range(epochs)
    x_val = range(1,epochs+1)
    
    plt.figure()
    plt.plot(x_train, hist.loss_train, 'r')
    plt.plot(x_val, hist.loss_val, 'k')
    plt.plot(x_val, hist.loss_val_image, 'b')
    plt.legend(['tot train', 'tot val', 'image val'], loc='upper right')
    
    ax = plt.gca()
    ax2=ax.twinx()
    ax2.plot(x_val, hist.loss_val_reg, 'm')
    ax2.legend(['reg val'], loc='lower right')
    plt.title('Training losses')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # lims = ax.get_ylim()
    # if lims[1] > 0.5:
    #     limsPlot = [lims[0], 0.5]
    # else:
    #     limsPlot = lims
    # ax.set_ylim(limsPlot)
    savePlot('trainPlot.png')

    if hist.model.param_activation:
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