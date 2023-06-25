import torch
import torch.nn as nn

class ConvGenerator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(ConvGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7,stride=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128,kernel_size= 4,stride= 2,padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, img_channels,kernel_size= 4,stride= 2,padding= 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        img = self.model(z)
        return img

class ConvDiscriminator(nn.Module):
    def __init__(self, img_channels):
        super(ConvDiscriminator, self).__init__()
        self.img_channels = img_channels

        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64,kernel_size= 4,stride= 2,padding= 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128,kernel_size= 4,stride= 2,padding= 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256,kernel_size= 4,stride= 2,padding= 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1,kernel_size= 3,stride= 1,padding= 0)
        )
        self.activate=nn.Sigmoid()
        # self.lay1=nn.Conv2d(img_channels, 64,kernel_size= 4,stride= 2,padding= 1)
        # self.lay2 = nn.LeakyReLU(0.2)
        # self.lay3 = nn.Conv2d(64, 128,kernel_size= 4,stride= 2,padding= 1)
        # self.lay4 = nn.BatchNorm2d(128)
        # self.lay5 = nn.LeakyReLU(0.2)
        # self.lay6 =nn.Conv2d(128, 256,kernel_size= 4,stride= 2,padding= 1)
        # self.lay7 = nn.BatchNorm2d(256)
        # self.lay8 = nn.LeakyReLU(0.2)
        # self.lay9 = nn.Conv2d(256, 1,kernel_size= 3,stride= 1,padding= 0)
        # self.lay10 = nn.Sigmoid()


    def forward(self, img):
        # output=self.lay1(img)
        # print(output.size())
        # output = self.lay2(output)
        # print(output.size())
        # output = self.lay3(output)
        # print(output.size())
        # output = self.lay4(output)
        # print(output.size())
        # output = self.lay5(output)
        # print(output.size())
        # output = self.lay6(output)
        # print(output.size())
        # output = self.lay7(output)
        # print(output.size())
        # output = self.lay8(output)
        # print(output.size())
        # output = self.lay9(output)
        # print(output.size())
        # output = self.lay10(output)
        # print(output.size())
        validity = self.model(img)
        validity = validity.view(-1,1)
        return self.activate(validity)


if __name__=='__main__':
    input=torch.randn(8,100)
    print(input.size())
    G=Generator(100,1)
    D=Discriminator(1)
    output=G(input)
    print(output.size())
    like=D(output)
    print(like)
