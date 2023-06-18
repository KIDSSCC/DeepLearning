import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    def __init__(self,in_dim,growth_rate):
        super(BottleNeck, self).__init__()
        inner_dim=4*growth_rate
        self.fc1=nn.BatchNorm2d(in_dim)
        self.fc2=nn.ReLU()
        self.fc3=nn.Conv2d(in_dim,inner_dim,kernel_size=1,bias=False)
        self.fc4=nn.BatchNorm2d(inner_dim)
        self.fc5=nn.ReLU()
        self.fc6=nn.Conv2d(inner_dim,growth_rate,kernel_size=3,bias=False)
    def forward(self,x):
        start=x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return torch.cat([start,x],1)


class Transition(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Transition, self).__init__()
        self.fc1=nn.BatchNorm2d(in_dim)
        self.fc2=nn.Conv2d(in_dim,out_dim,1,bias=False)
        self.fc3=nn.AvgPool2d(2,stride=2)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class DenseNet(nn.Module):
    def __init__(self,block,nblocks,growth_rate=12,reduction=0.5,num_class=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        inner_dim=2*self.growth_rate

        self.fc1=nn.Conv2d(3, inner_dim, kernel_size=3, padding=1, bias=False)
        self.fc2=nn.Sequential()
        for i in range(len(nblocks)-1):
            self.fc2.add_module("dense_block_layer_{}".format(i),self.make_layer(block, inner_dim, nblocks[i]))
            inner_dim += growth_rate * nblocks[i]
            out_dim = int(reduction * inner_dim)
            self.fc2.add_module("transition_layer_{}".format(i), Transition(inner_dim, out_dim))
            inner_dim = out_dim

        self.fc2.add_module("dense_block{}".format(len(nblocks) - 1),self.make_layer(block, inner_dim, nblocks[len(nblocks) - 1]))
        inner_dim += growth_rate * nblocks[len(nblocks) - 1]

        self.fc2.add_module('bn', nn.BatchNorm2d(inner_dim))
        self.fc2.add_module('relu', nn.ReLU(inplace=True))

        self.fc3 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc4 = nn.Linear(inner_dim, num_class)

    def make_layer(self,block,in_dim,nblocks):
        dense_block=nn.Sequential()
        for i in range(nblocks):
            dense_block.add_module('bottle_Neck',block(in_dim,self.growth_rate))
            in_dim += self.growth_rate
        return dense_block

    def forward(self,x):
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        output = output.view(output.size()[0], -1)
        return self.fc4(output)

def densenet121():
    return DenseNet(BottleNeck, [6,12,24,16], growth_rate=32)






