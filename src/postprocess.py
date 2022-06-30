import matplotlib.pyplot as plt
import torch
from src.image_naming import getParamsFromImageName, getMusFromParams

def plotImage(data, nRows, n_disp, count):
        ax = plt.subplot(nRows, n_disp, count)
        plt.imshow(data)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

def plotting(input, code, pred, img_names):
    nRows = 4
    plotNames = ['X', 'code', 'X_NN', 'fig data']
    n_disp = len(input)
    plt.figure(figsize=(20, 5))
    for i in range(n_disp):
        plotImage(input[i], nRows, n_disp, i+1)
        plotImage(code[i], nRows, n_disp, i+1+n_disp)
        plotImage(pred[i], nRows, n_disp, i+1+2*n_disp)

        # Display error for each test image
        mu1, mu2 = getMusFromImgName(img_names[i])
        img_error = torch.mean((pred[i]-input[i,:,:,0])**2)

        imageError = '{:.2}'.format(img_error.item())
        ax = plt.subplot(4, n_disp, i + 1 + 3*n_disp)
        ax.text(0.15,0.5,'mu1 = {}'.format(mu1))
        ax.text(0.15,0.3,'mu2 = {}'.format(mu2))
        ax.text(0.15,0.1,'loss = {}'.format(imageError))
        ax.axis('off')
    addPlotNames(plotNames)
    plt.show()

def getMusFromImgName(imgName):
        Fh, Fv, loc, pos = getParamsFromImageName(imgName)
        return getMusFromParams(Fh, Fv, loc, pos)

def addPlotNames(plotNames):
    for i, plotName in enumerate(reversed(plotNames)):
        plt.text(0.1, 0.12+0.23*i, plotName, fontsize=12, transform=plt.gcf().transFigure, rotation=90)
