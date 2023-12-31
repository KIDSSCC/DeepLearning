import sys
print(sys.version) # python 3.6
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
print(torch.__version__) # 1.0.1

import matplotlib.pyplot as plt


dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))]),
                       download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1) # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())


class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = x.view(x.size(0), 784) # flatten (bs x 1 x 28 x 28) -> (bs x 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)
    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out) # range [-1, 1]
        # convert to image
        out = out.view(out.size(0), 1, 28, 28)
        return out


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)

from ConvGAN import *
D=ConvDiscriminator(1).to(device)
G=ConvGenerator(100,1).to(device)
# D = Discriminator().to(device)
# G = Generator().to(device)

# optimizerD = torch.optim.SGD(D.parameters(), lr=0.03)
# optimizerG = torch.optim.SGD(G.parameters(), lr=0.03)
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002)
optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002)

criterion = nn.BCELoss()

lab_real = torch.ones(64, 1, device=device)
lab_fake = torch.zeros(64, 1, device=device)

D_allloss=[]
G_allloss=[]
D_run_loss=0
G_run_loss=0

for epoch in range(3):  # 3 epochs
    for i, data in enumerate(dataloader, 0):
        # STEP 1: Discriminator optimization step
        x_real, _ = next(iter(dataloader))
        x_real = x_real.to(device)
        optimizerD.zero_grad()

        D_x = D(x_real)
        lossD_real = criterion(D_x, lab_real)

        z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
        x_gen = G(z).detach()
        D_G_z = D(x_gen)
        lossD_fake = criterion(D_G_z, lab_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # STEP 2: Generator optimization step
        optimizerG.zero_grad()

        z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
        x_gen = G(z)
        D_G_z = D(x_gen)
        lossG = criterion(D_G_z, lab_real)  # -log D(G(z))

        lossG.backward()
        optimizerG.step()
        if i % 100 == 0:
            D_allloss.append(lossD)
            G_allloss.append(lossG)
            print('epoch:{}.index:{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                epoch, i, len(dataloader), D_x.mean().item(), D_G_z.mean().item()))

for i in range(len(D_allloss)):
    D_allloss[i]=D_allloss[i].detach()
for i in range(len(G_allloss)):
    G_allloss[i]=G_allloss[i].detach()
print(D_allloss)
print(G_allloss)
plt.figure()
plt.plot(D_allloss)
plt.title('D_Loss')

plt.figure()
plt.plot(G_allloss)
plt.title('G_Loss')
plt.show()

inputs=torch.randn(8,100)
out=G(inputs)
show_imgs(out)
# change 1
clone1=inputs
clone1[:,0]=0.1
out1=G(clone1)
clone1[:,0]=3
out2=G(clone1)
clone1[:,0]=10
out3=G(clone1)
outputs=torch.cat([out1,out2,out3],dim=0)
show_imgs(outputs)
# change 2
clone2=inputs
clone2[:,10]=-10
out1=G(clone2)
clone2[:,10]=0
out2=G(clone2)
clone2[:,10]=10
out3=G(clone2)
outputs=torch.cat([out1,out2,out3],dim=0)
show_imgs(outputs)
# change 3
clone3=inputs
clone3[:,49]=-10
out1=G(clone3)
clone3[:,49]=0
out2=G(clone3)
clone3[:,49]=10
out3=G(clone3)
outputs=torch.cat([out1,out2,out3],dim=0)
show_imgs(outputs)
# change 4
clone4=inputs
clone4[:,75]=-10
out1=G(clone4)
clone4[:,75]=0
out2=G(clone4)
clone4[:,75]=10
out3=G(clone4)
outputs=torch.cat([out1,out2,out3],dim=0)
show_imgs(outputs)
# change 5
clone5=inputs
clone5[:,99]=-20
out1=G(clone5)
clone5[:,99]=0
out2=G(clone5)
clone5[:,99]=20
out3=G(clone5)
outputs=torch.cat([out1,out2,out3],dim=0)
show_imgs(outputs)





