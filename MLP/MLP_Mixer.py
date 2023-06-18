import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange,Reduce

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)
batch_size = 32
train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())
validation_dataset = datasets.MNIST('./data',
                                    train=False,
                                    transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)


def token_Mixer(dim,convdim,expansion_factor=4):
    inner_dim=dim*expansion_factor
    dropout=0.1
    block=nn.Sequential(
        nn.LayerNorm(dim),
        nn.Conv1d(convdim,inner_dim,kernel_size=1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(inner_dim, convdim,kernel_size=1),
        nn.Dropout(dropout)
    )
    return block

def channel_Mixer(dim,expansion_factor=4):
    inner_dim = dim * expansion_factor
    dropout = 0.1
    block = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    )
    return block

class MLP_Mixer(nn.Module):
    def __init__(self,patch_size,depth):
        super(MLP_Mixer, self).__init__()
        self.fc1=Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.fc2=nn.Linear((patch_size ** 2), 100)
        self.fc3=list()
        for i in range(depth):
            self.fc3.append(token_Mixer(100,16))
            self.fc3.append(channel_Mixer(100))
        self.fc4=nn.LayerNorm(100)
        self.fc5=Reduce('b n c -> b c', 'mean')
        self.fc6=nn.Linear(100, 10)

    def forward(self,x):
        # print('before fc1,the size is '+str(x.size()))
        x=self.fc1(x)
        # print('before fc2,the size is ' + str(x.size()))
        x=self.fc2(x)
        # print('before fc3,the size is ' + str(x.size()))
        for layer in self.fc3:
            res=x
            x=res+layer(x)
        # print('before fc4,the size is ' + str(x.size()))
        x=self.fc4(x)
        # print('before fc5,the size is ' + str(x.size()))
        x=self.fc5(x)
        # print('before fc6,the size is ' + str(x.size()))
        return F.log_softmax(self.fc6(x),dim=1)


model = MLP_Mixer(7,4).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)

def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()  # w - alpha * dL / dw
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))


def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))



epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)



