import numpy as np
import matplotlib.pyplot as plt
from src.postprocess import savePlot


class Elipse():

    def getMusFromImgName(self, name):
        return [float(name[name.find('mu')+2:name.find('png')-1])]

    def getMuDomain(self):
            return np.linspace(0.5, 2, 2700)

    def plotDataset(self, mus_test_ext):
        mu = self.getMuDomain()
        mu_test = mus_test_ext[0]

        # Plot points for training and testing
        fig, ax = plt.subplots()
        ax.scatter(mu, np.zeros(len(mu)), color='blue')
        ax.scatter(mu_test, np.zeros(len(mu_test)), color='red')
        plt.xlabel("mu = a/b (elipse ratio)")
        savePlot('datasetPlot.png')