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

def plotting(input, code, pred, code_trunc, pred_trunc, img_names, zero_code, trunc_code, data_class):
    nRows = 6
    plotNames = ['X', 'code', 'X_NN', 'code trunc', 'X_NN trunc', 'fig data']
    n_disp = len(input)
    plt.figure(figsize=(20, 8))
    for i in range(n_disp):
        plotImage(input[i], nRows, n_disp, i+1)
        plotImage(code[i], nRows, n_disp, i+1+n_disp)
        plotZeroCode(len(code.data[0]), zero_code, 'blue')
        plotImage(pred[i], nRows, n_disp, i+1+2*n_disp)
        plotImage(code_trunc[i], nRows, n_disp, i+1+3*n_disp)
        plotZeroCode(len(code.data[0]), zero_code, 'blue')
        plotZeroCode(len(code.data[0]), trunc_code, 'red')
        plotImage(pred_trunc[i], nRows, n_disp, i+1+4*n_disp)

        # Display error for each test image
        mus = data_class.getMusFromImgName(img_names[i])
        img_error = torch.mean((pred[i]-input[i,:,:,0])**2)

        imageError = '{:.2}'.format(img_error.item())
        ax = plt.subplot(nRows, n_disp, i + 1 + 5*n_disp)
        for imu in range(len(mus)):
            ax.text(0.15,0.5-(imu*0.2),'mu{} = {}'.format(imu+1, mus[imu]))
        ax.text(0.15,0.1,'loss = {}'.format(imageError))
        ax.axis('off')
    addPlotNames(plotNames)
    savePlot('predictsPlot.png')

def addPlotNames(plotNames):
    for i, plotName in enumerate(reversed(plotNames)):
        plt.text(0.1, 0.12+0.14*i, plotName, fontsize=12, transform=plt.gcf().transFigure, rotation=90)

def plotTraining(epochs, loss_train, loss_val, alphas):
    plt.figure()
    plt.plot(range(epochs), loss_train)
    plt.plot(range(1,epochs+1), loss_val)
    plt.title('Training losses')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    ax = plt.gca()
    lims = ax.get_ylim()
    if lims[1] > 0.5:
        limsPlot = [lims[0], 0.5]
    else:
        limsPlot = lims
    ax.set_ylim(limsPlot)
    savePlot('trainPlot.png')

    plt.figure()
    plt.plot(range(epochs), alphas[0])
    plt.plot(range(epochs), alphas[1])
    plt.title('Activation function parameters')
    plt.ylabel('alpha')
    plt.xlabel('epoch')
    plt.legend(['alpha ReLu', 'alpha Sigmoid'], loc='upper right')
    savePlot('alphasPlot.png')

def savePlot(name):
    plt.savefig('results/{}'.format(name))

def plotShow():
    plt.show()