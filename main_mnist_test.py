import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def plotImage(data, nRows, numDisplay, count):
    ax = plt.subplot(nRows, numDisplay, count)
    plt.imshow(data)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def train(dataloader, model, loss_fn, optimizer, reg):
    size = len(dataloader.dataset)

    for batch, (X, _) in enumerate(dataloader):
        # Compute prediction and loss
        pred, code = model(X)
        pred = torch.reshape(pred, (X.shape[0], 1, 28, 28))
        loss = loss_fn(pred, X) + reg*torch.mean(torch.abs(code))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, reg):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, _ in dataloader:
            pred, code = model(X)
            pred = torch.reshape(pred, (X.shape[0], 1, 28, 28))
            test_loss += loss_fn(pred, X).item() + reg*torch.mean(torch.abs(code))

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


class Autoencoder(nn.Module):
    def __init__(self, codeSize):
        super(Autoencoder, self).__init__()
        self.codeSize = codeSize

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, self.codeSize),
            nn.ReLU(),
            nn.Linear(self.codeSize, self.codeSize),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.codeSize, self.codeSize),
            nn.ReLU(),
            nn.Linear(self.codeSize, 28*28),
        )

    def forward(self, x):
        code = self.encoder(x)
        x_nn = self.decoder(code)
        return x_nn, code


# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5
codeSize = 36
reg = 1e-4

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

loss_fn = nn.MSELoss()
model = Autoencoder(codeSize)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, reg)
    test(test_dataloader, model, loss_fn, reg)
print("Training finished!")

# Make predictions
nDisp = 6
val_data = test_data.data[:nDisp,:,:]
val_data = torch.reshape(val_data, (val_data.shape[0], 1, 28, 28))
val_data = torch.from_numpy(np.ascontiguousarray(val_data)/np.array(255, dtype=np.float32))

with torch.no_grad():
    pred, code = model(val_data)

# Detect dimensionality
latent = np.sum(code.numpy()**2, axis=0)**0.5
rel_latent = latent/np.max(latent)

active_code = np.zeros(codeSize, dtype=int)

for i in range(codeSize):
    if rel_latent[i] > 0.1: active_code[i] = 1

latent_trunc_dim = sum(active_code)

code = torch.reshape(code, (code.shape[0], int(np.sqrt(codeSize)), int(np.sqrt(codeSize))))
pred = torch.reshape(pred, (pred.shape[0], 28, 28))

nRows = 6
plt.figure(figsize=(20, 5))
for i in range(nDisp):
    plotImage(test_data.data[i], nRows, nDisp, i+1)
    plotImage(code[i], nRows, nDisp, i+1+nDisp)
    plotImage(pred[i], nRows, nDisp, i+1+2*nDisp)
plt.show()
test = 1
