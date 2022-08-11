import numpy as np
import matplotlib.pyplot as plt
from src.postprocess import savePlot

class BeamHomog():

    def getMusFromImgName(self, imgName):
            Fh, Fv, loc, pos = self.getParamsFromImageName(imgName)
            return self.getMusFromParams(Fh, Fv, loc, pos)

    def getMusFromParams(self, Fh, Fv, loc, pos):
            if Fh == 0 and Fv == 1:
                mu2 = 0
            elif Fh == 0.052336 and Fv == 0.99863:
                mu2 = 3
            elif Fh == 0.104528 and Fv == 0.994522:
                mu2 = 6
            elif Fh == 0.156434 and Fv == 0.987688:
                mu2 = 9
            elif Fh == 0.207912 and Fv == 0.978148:
                mu2 = 12
            elif Fh == 0.258819 and Fv == 0.965926:
                mu2 = 15
            elif Fh == 0.309017 and Fv == 0.951057:
                mu2 = 18
            elif Fh == 0.358368 and Fv == 0.93358:
                mu2 = 21
            elif Fh == 0.406737 and Fv == 0.913545:
                mu2 = 24
            elif Fh == 0.45399 and Fv == 0.891007:
                mu2 = 27
            elif Fh == 0.5 and Fv == 0.866025:
                mu2 = 30
            elif Fh == 0.544639 and Fv == 0.838671:
                mu2 = 33
            elif Fh == 0.587785 and Fv == 0.809017:
                mu2 = 36
            elif Fh == 0.62932 and Fv == 0.777146:
                mu2 = 39
            elif Fh == 0.669131 and Fv == 0.743145:
                mu2 = 42
            elif Fh == 0.707107 and Fv == 0.707107:
                mu2 = 45
            elif Fh == 0.743145 and Fv == 0.669131:
                mu2 = 48
            elif Fh == 0.777146 and Fv == 0.62932:
                mu2 = 51
            elif Fh == 0.809017 and Fv == 0.587785:
                mu2 = 54
            elif Fh == 0.838671 and Fv == 0.544639:
                mu2 = 57
            elif Fh == 0.866025 and Fv == 0.5:
                mu2 = 60
            elif Fh == 0.891007 and Fv == 0.45399:
                mu2 = 63
            elif Fh == 0.913545 and Fv == 0.406737:
                mu2 = 66
            elif Fh == 0.93358 and Fv == 0.358368:
                mu2 = 69
            elif Fh == 0.951057 and Fv == 0.309017:
                mu2 = 72
            elif Fh == 0.965926 and Fv == 0.258819:
                mu2 = 75
            elif Fh == 0.978148 and Fv == 0.207912:
                mu2 = 78
            elif Fh == 0.987688 and Fv == 0.156434:
                mu2 = 81
            elif Fh == 0.994522 and Fv == 0.104528:
                mu2 = 84
            elif Fh == 0.99863 and Fv == 0.052336:
                mu2 = 87
            elif Fh == 1 and Fv == 0:
                mu2 = 90
            elif Fh == 0.99863 and Fv == -0.052336:
                mu2 = 93
            elif Fh == 0.994522 and Fv == -0.104528:
                mu2 = 96
            elif Fh == 0.987688 and Fv == -0.156434:
                mu2 = 99
            elif Fh == 0.978148 and Fv == -0.207912:
                mu2 = 102
            elif Fh == 0.965926 and Fv == -0.258819:
                mu2 = 105
            elif Fh == 0.951057 and Fv == -0.309017:
                mu2 = 108
            elif Fh == 0.93358 and Fv == -0.358368:
                mu2 = 111
            elif Fh == 0.913545 and Fv == -0.406737:
                mu2 = 114
            elif Fh == 0.891007 and Fv == -0.45399:
                mu2 = 117
            elif Fh == 0.866025 and Fv == -0.5:
                mu2 = 120
            elif Fh == 0.838671 and Fv == -0.544639:
                mu2 = 123
            elif Fh == 0.809017 and Fv == -0.587785:
                mu2 = 126
            elif Fh == 0.777146 and Fv == -0.62932:
                mu2 = 129
            elif Fh == 0.743145 and Fv == -0.669131:
                mu2 = 132
            elif Fh == 0.707107 and Fv == -0.707107:
                mu2 = 135
            elif Fh == 0.669131 and Fv == -0.743145:
                mu2 = 138
            elif Fh == 0.62932 and Fv == -0.777146:
                mu2 = 141
            elif Fh == 0.587785 and Fv == -0.809017:
                mu2 = 144
            elif Fh == 0.544639 and Fv == -0.838671:
                mu2 = 147
            elif Fh == 0.5 and Fv == -0.866025:
                mu2 = 150
            elif Fh == 0.45399 and Fv == -0.891007:
                mu2 = 153
            elif Fh == 0.406737 and Fv == -0.913545:
                mu2 = 156
            elif Fh == 0.358368 and Fv == -0.93358:
                mu2 = 159
            elif Fh == 0.309017 and Fv == -0.951057:
                mu2 = 162
            elif Fh == 0.258819 and Fv == -0.965926:
                mu2 = 165
            elif Fh == 0.207912 and Fv == -0.978148:
                mu2 = 168
            elif Fh == 0.156434 and Fv == -0.987688:
                mu2 = 171
            elif Fh == 0.104528 and Fv == -0.994522:
                mu2 = 174
            elif Fh == 0.052336 and Fv == -0.99863:
                mu2 = 177
            else:
                raise ValueError('requested mu2 = {} not available in dataset'.format(mu2))
            
            if loc == 'B':
                mu1 = pos
            elif loc == 'R':
                mu1 = pos + 1
            elif loc == 'T':
                mu1 = 3 - pos

            return [mu1, mu2]

    def getParamsFromMus(self, mu1, mu2):
            if mu2 == 0:
                Fh = 0; Fv = 1
            elif mu2 == 3: 
                Fh = 0.052336 ; Fv = 0.99863
            elif mu2 == 6:
                Fh = 0.104528; Fv = 0.994522
            elif mu2 == 9:
                Fh = 0.156434; Fv = 0.987688
            elif mu2 == 12:
                Fh = 0.207912; Fv = 0.978148
            elif mu2 == 15: 
                Fh = 0.258819; Fv = 0.965926
            elif mu2 == 18: 
                Fh = 0.309017; Fv = 0.951057
            elif mu2 == 21: 
                Fh = 0.358368; Fv = 0.93358
            elif mu2 == 24:
                Fh = 0.406737; Fv = 0.913545
            elif mu2 == 27: 
                Fh = 0.45399; Fv = 0.891007
            elif mu2 == 30: 
                Fh = 0.5 ; Fv = 0.866025
            elif mu2 == 33: 
                Fh = 0.544639; Fv = 0.838671
            elif mu2 == 36: 
                Fh = 0.587785; Fv = 0.809017
            elif mu2 == 39: 
                Fh = 0.62932 ; Fv = 0.777146
            elif mu2 == 42:
                Fh = 0.669131; Fv = 0.743145
            elif mu2 == 45:
                Fh = 0.707107; Fv = 0.707107
            elif mu2 == 48:
                Fh = 0.743145; Fv = 0.669131
            elif mu2 == 51:
                Fh = 0.777146; Fv = 0.62932
            elif mu2 == 54:
                Fh = 0.809017; Fv = 0.587785
            elif mu2 == 57:
                Fh = 0.838671; Fv = 0.544639
            elif mu2 == 60:
                Fh = 0.866025; Fv = 0.5
            elif mu2 == 63:
                Fh = 0.891007; Fv = 0.45399
            elif mu2 == 66:
                Fh = 0.913545; Fv = 0.406737
            elif mu2 == 69:
                Fh = 0.93358; Fv = 0.358368
            elif mu2 == 72:
                Fh = 0.951057; Fv = 0.309017
            elif mu2 == 75:
                Fh = 0.965926; Fv = 0.258819
            elif mu2 == 78:
                Fh = 0.978148; Fv = 0.207912
            elif mu2 == 81:
                Fh = 0.987688; Fv = 0.156434
            elif mu2 == 84:
                Fh = 0.994522; Fv = 0.104528
            elif mu2 == 87:
                Fh = 0.99863; Fv = 0.052336
            elif mu2 == 90:
                Fh = 1; Fv = 0
            elif mu2 == 93:
                Fh = 0.99863; Fv = -0.052336
            elif mu2 == 96:
                Fh = 0.994522; Fv = -0.104528
            elif mu2 == 99: 
                Fh = 0.987688; Fv = -0.156434
            elif mu2 == 102:
                Fh = 0.978148; Fv = -0.207912
            elif mu2 == 105:
                Fh = 0.965926; Fv = -0.258819
            elif mu2 == 108:
                Fh = 0.951057; Fv = -0.309017
            elif mu2 == 111:
                Fh = 0.93358; Fv = -0.358368
            elif mu2 == 114:
                Fh = 0.913545; Fv = -0.406737
            elif mu2 == 117:
                Fh = 0.891007; Fv = -0.45399
            elif mu2 == 120:
                Fh = 0.866025; Fv = -0.5
            elif mu2 == 123: 
                Fh = 0.838671; Fv = -0.544639
            elif mu2 == 126: 
                Fh = 0.809017; Fv = -0.587785
            elif mu2 == 129:
                Fh = 0.777146; Fv = -0.62932
            elif mu2 == 132:
                Fh = 0.743145; Fv = -0.669131
            elif mu2 == 135:
                Fh = 0.707107; Fv = -0.707107
            elif mu2 == 138:
                Fh = 0.669131; Fv = -0.743145
            elif mu2 == 141:
                Fh = 0.62932; Fv = -0.777146
            elif mu2 == 144:
                Fh = 0.587785; Fv = -0.809017
            elif mu2 == 147:
                Fh = 0.544639; Fv = -0.838671
            elif mu2 == 150:
                Fh = 0.5; Fv = -0.866025
            elif mu2 == 153:
                Fh = 0.45399; Fv = -0.891007
            elif mu2 == 156:
                Fh = 0.406737; Fv = -0.913545
            elif mu2 == 159:
                Fh = 0.358368; Fv = -0.93358
            elif mu2 == 162:
                Fh = 0.309017; Fv = -0.951057
            elif mu2 == 165:
                Fh = 0.258819; Fv = -0.965926
            elif mu2 == 168:
                Fh = 0.207912; Fv = -0.978148
            elif mu2 == 171:
                Fh = 0.156434; Fv = -0.987688
            elif mu2 == 174:
                Fh = 0.104528; Fv = -0.994522
            elif mu2 == 177:
                Fh = 0.052336; Fv = -0.99863           
            else:
                raise ValueError('requested mu2 = {} not available in dataset'.format(mu2))

            if mu1 < 1.0:
                loc = 'B'
                pos = mu1
            elif 1.0 <= mu1 < 2.0:
                loc = 'R'
                pos = mu1 - 1
            elif 2.0 <= mu1:
                loc = 'T'
                pos = 1 - (mu1 - 2)

            pos = round(pos, 2)
            if pos == 0.0:
                pos = 0

            return Fh, Fv, loc, pos

    def getMuDomain(self):
            mu2 = [round(x,2) for x in np.arange(0, 180, 3)]
            muBot = [round(x,2) for x in np.arange(0.3, 0.95, 0.05)]
            muRight = [round(x,2) for x in np.arange(0, 0.95, 0.05)]
            muTop = muBot

            mu1 = list(muBot)
            for locRight in muRight:
                mu1.append(locRight+1)
            
            for locTop in reversed(muTop):
                mu1.append(3 - locTop)

            mus_ext = np.meshgrid(mu1, mu2)
            return mu1, mu2, mus_ext[0], mus_ext[1]

    def getParamsFromImageName(self, name):
            # Get indexes of the two underscores
            underscores = [i for i, ltr in enumerate(name) if ltr == '_']
            assert len(underscores) == 2, 'Image name with incorrect name. It must contain exactly 2 underscores characters'
            Fh = float(name[2:underscores[0]])
            Fv = float(name[underscores[0]+3:underscores[1]])
            loc = name[underscores[1]+1:underscores[1]+2]
            pos = float(name[underscores[1]+2:-4])
            return Fh, Fv, loc, pos

    def getImageNamesFromMus(self, mus_test):
            mu1_ext = []
            mu2_ext = []
            testData = []
            for mu1 in mus_test[0]:
                for mu2 in mus_test[1]:
                    mu1_ext.append(mu1)
                    mu2_ext.append(mu2)
                    Fh, Fv, loc, pos = self.getParamsFromMus(mu1,mu2)
                    name = 'Fh{}_Fv{}_{}{}.txt'.format(Fh, Fv, loc, pos)
                    testData.append(name)
            return testData, mu1_ext, mu2_ext

    def plotDataset(self, mus_test_ext, fig_path, name):
        mu1, mu2, mu1_ext, mu2_ext = self.getMuDomain()
        mu1_test = mus_test_ext[0]
        mu2_test = mus_test_ext[1]

        # Plot points for training and testing
        fig, ax = plt.subplots()
        ax.scatter(mu1_ext, mu2_ext, color='blue')
        ax.scatter(mu1_test, mu2_test, color='red')
        ax.set_xticks(mu1)
        ax.set_yticks(mu2)
        plt.xlabel("mu_1 (position)")
        plt.ylabel("mu_2 (angle in º)")
        savePlot(f'{fig_path}/datasetPlot_{name}.png')