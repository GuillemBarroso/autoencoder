from torch.utils.data import Dataset
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class DataTorch(Dataset):
    def __init__(self, data):
        self.data = data
        self.resolution = data.shape[1:]
        self.dimension = np.prod(self.resolution[0:2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Data(Dataset):
    def __init__(self, dataset, testSize=0.1):
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.dimension = None
        self.resolution = None
        self.imgNames = None
        self.imgTestNames = None

        self.testSize = testSize
        self.dataset = dataset

        data = self.__getData()

        x_train, x_val, x_test = self.__splitData(data)
        self.__storeDataForTraining(x_train, x_val, x_test)


    def __getData(self):
        path = 'data/{}/data.npz'.format(self.dataset)
        if not os.path.exists(path):
            self.mesh = Mesh(self.dataset)
            data, self.imgNames = self.__readFiles()
            self.__saveDataAsNumpy(path, data)
        else:
            file = np.load(path)
            data = file['data']
            self.imgNames = file['imgNames']
        
        return data

    def __readFiles(self):
        # Read results for all cases
        files = os.listdir(self.mesh.dir)
        data = []
        names = []
        for count, file in enumerate(files):
            if count % 100 == 0:
                print('Reading file {}/{}'.format(count, len(files)))
            if file[-3:] == 'txt':
                values = self.__extractValuesFromFile(file)
                pixels = self.__computePixelsFromP0values(values)
                data.append(pixels)
                names.append(file)

        return np.asarray(data), np.asarray(names)

    def __extractValuesFromFile(self, file):
        aux = open('{}/{}'.format(self.mesh.dir, file),"r")
        lines = aux.readlines()
        values = []
        for line in lines:
            lineSep = getStringsTabSeparated(line)
            for val in lineSep:
                if val:
                    newVal = float(val)
                    values.append(newVal)

        nElemsAux = values[0]
        assert self.mesh.nElems == nElemsAux, 'Different number of elements when reading nodal values from TXT files'
        aux.close()
        return values[1:]
    
    def __computePixelsFromP0values(self, values):
        # Create data array with pixel values combining element values in the same pixel
        # Assume ordering of elements coming from FreeFEM and structured mesh 
        # (first row first with all elements ordered from left to right)
        data = np.asarray([ [0]*self.mesh.nElemsX for i in range(self.mesh.nElemsY)], dtype=float).reshape(self.mesh.nElemsY, self.mesh.nElemsX, 1)
                
        count = 0
        for i in range(self.mesh.nElemsY):
            for j in range(self.mesh.nElemsX):
                data[-i,j] = 0.5*(values[count] + values[count+1])
                count += 2

        return data

    def __saveDataAsNumpy(self, path, data):
        np.savez(path, data=data, imgNames=self.imgNames)

    def __splitData(self, data):
        valSize = self.testSize/(1-self.testSize)
        idx = range(len(data))
        x_train, x_test, _, idx_test = train_test_split(data, idx, test_size=self.testSize, shuffle=True)
        x_train, x_val = train_test_split(x_train, test_size=valSize, shuffle=True)
        self.imgTestNames = [self.imgNames[x] for x in idx_test]

        return x_train, x_val, x_test

    def __storeDataForTraining(self, x_train, x_val, x_test):
        # Store training data in correct format
        self.x_train = torch.from_numpy(x_train).float()
        self.x_val = torch.from_numpy(x_val).float()
        self.x_test = torch.from_numpy(x_test).float()

        # Store extra info
        self.resolution = self.x_train.shape[1:]
        self.dimension = np.prod(self.resolution[0:2])
        self.nTrain = self.x_train.shape[0]
        self.nVal = self.x_val.shape[0]
        self.nTest = self.x_test.shape[0]


def getStringsSpaceSeparated(line):
    return line[:-1].split(" ")

def getStringsTabSeparated(line):
    return line[:-1].split("\t")

class Mesh():
    def __init__(self, dataset):
        self.nodes = None
        self.elems = None
        self.edges = None
        self.dir = None
        self.nElems = None
        self.nNodes = None
        self.nBoundEdges = None
        self.nElemsX = -1
        self.nElemsY = -1
        self.__getMesh__(dataset)

    def __getMesh__(self, dataset):
        # getMesh assumes that the mesh file is called Th.msh and that contains
        # data separated with spaces

        # Read mesh
        name = 'Th.msh'
        self.dir = r'data/{}'.format(dataset)
        mesh = open('{}/{}'.format(self.dir, name),"r")
        lines = mesh.readlines()
        line = getStringsSpaceSeparated(lines[0])

        self.nNodes, self.nElems, self.nBoundEdges = int(line[0]), int(line[1]), int(line[2])

        # Loop on nodes
        # nodes = [[x, y, boundID]_1, [x, y, boundID]_2, ...]
        self.nodes = []
        for i in range(1, self.nNodes+1):
            line = getStringsSpaceSeparated(lines[i])
            newLine = [float(line[0]), float(line[1]), int(line[2])]
            self.nodes.append(newLine)

            # Find number of elements in x and y directions
            if newLine[1] == 0.0:
                self.nElemsX += 1
            if newLine[0] == 1.0:
                self.nElemsY += 1

        # Loop on elements
        # elems = [[n1, n2, n3]_1, [n1, n2, n3]_2, ...]
        self.elems = []
        for i in range(self.nNodes+1, self.nNodes+self.nElems+1):
            line = getStringsSpaceSeparated(lines[i])
            newLine = [int(line[0]), int(line[1]), int(line[2])]
            self.elems.append(newLine)

        # Loop on edges
        # edges = [[n1, n2, boundID]_1, [n1, n2, boundID]_2, ...]
        self.edges = []
        for i in range(self.nNodes+self.nElems+1, self.nNodes+self.nElems+self.nBoundEdges+1):
            line = getStringsSpaceSeparated(lines[i])
            newLine = [int(line[0]), int(line[1]), int(line[2])]
            self.edges.append(newLine)

        mesh.close()