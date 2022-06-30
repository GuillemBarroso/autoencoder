def getMusFromParams(Fh, Fv, loc, pos):
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

        return mu1, mu2


def getParamsFromImageName(name):
        # Get indexes of the two underscores
        underscores = [i for i, ltr in enumerate(name) if ltr == '_']
        assert len(underscores) == 2, 'Image name with incorrect name. It must contain exactly 2 underscores characters'
        Fh = float(name[2:underscores[0]])
        Fv = float(name[underscores[0]+3:underscores[1]])
        loc = name[underscores[1]+1:underscores[1]+2]
        pos = float(name[underscores[1]+2:-4])
        return Fh, Fv, loc, pos