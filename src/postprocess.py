import matplotlib.pyplot as plt


def plotImage(data, nRows, numDisplay, count):
        ax = plt.subplot(nRows, numDisplay, count)
        plt.imshow(data)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

def plotting(n_disp, input, code, pred):
    if n_disp > len(input): n_disp = len(input)
    nRows = 3
    plt.figure(figsize=(20, 5))
    for i in range(n_disp):
        plotImage(input[i], nRows, n_disp, i+1)
        plotImage(code[i], nRows, n_disp, i+1+n_disp)
        plotImage(pred[i], nRows, n_disp, i+1+2*n_disp)
    plt.show()