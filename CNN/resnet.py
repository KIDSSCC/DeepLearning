import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,expansion,stride,downsample=False):
        super(BasicBlock, self).__init__()
        self.expansion=expansion
        # 残差块
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion)
        )
        # 残差连接
        self.shortcut = nn.Sequential()
        if downsample:
            # 进行下采样，对齐维度
            self.shortcut.add_module("conv",nn.Conv2d(in_channel,out_channel*self.expansion,kernel_size=1,stride=stride,bias=False))
            self.shortcut.add_module("BN",nn.BatchNorm2d(out_channel*self.expansion))

    def forward(self, input):
        output=self.residual_function(input)+self.shortcut(input)
        return nn.ReLU(inplace=True)(output)

# class BasicBlock(nn.Module):
#     expansion=1
#     def __init__(self,in_channels,out_channels,stride=1):
#         super().__init__()
#         # 残差块
#         self.residual_function=nn.Sequential(
#             nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels,out_channels*BasicBlock.expansion,kernel_size=3,padding=1,bias=False),
#             nn.BatchNorm2d(out_channels*BasicBlock.expansion)
#         )
#         # 短路连接,发生了下采样或者输入通道数不等于输出通道数
#         self.shortcut = nn.Sequential()
#         if stride!=1 or in_channels!=out_channels*BasicBlock.expansion:
#             self.shortcut=nn.Sequential(
#                 nn.Conv2d(in_channels,out_channels*BasicBlock.expansion,kernel_size=1,stride=stride,bias=False),
#                 nn.BatchNorm2d(out_channels*BasicBlock.expansion)
#             )
#     def forward(self,x):
#         # 最终返回的通道数为out_channels*BasicBlock.expansion
#         return nn.ReLU(inplace=True)(self.residual_function(x)+self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()

        self.in_channels=64
        self.expansion=1
        # c1阶段：由3通道变为64通道，图像大小不发生改变
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # c2阶段：
        self.conv2=nn.Sequential(
            BasicBlock(64,64,self.expansion,1),
            BasicBlock(64*self.expansion,64,self.expansion,1)
        )
        # c3阶段：
        self.conv3=nn.Sequential(
            BasicBlock(64,128,self.expansion,2,True),
            BasicBlock(128*self.expansion,128,self.expansion,1),
        )
        # c4阶段：
        self.conv4=nn.Sequential(
            BasicBlock(128, 256, self.expansion, 2, True),
            BasicBlock(256*self.expansion, 256, self.expansion, 1),
        )
        # c5阶段
        self.conv5=nn.Sequential(
            BasicBlock(256, 512, self.expansion, 2, True),
            BasicBlock(512*self.expansion, 512, self.expansion, 1),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)



        # self.conv2_x = self._make_layer(BasicBlock,64,2,1)
        # self.conv3_x = self._make_layer(BasicBlock, 128, 2, 2)
        # self.conv4_x = self._make_layer(BasicBlock, 256, 2, 2)
        # self.conv5_x = self._make_layer(BasicBlock, 512, 2, 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc=nn.Linear(512*self.expansion,num_classes)



    # def _make_layer(self,block,out_channels,num_blocks,stride):
    #     # c2至c5,分别是[1,1],[2,1],[2,1],[2,1],
    #     strides=[stride]+[1]*(num_blocks-1)
    #     layers=[]
    #     for stride in strides:
    #         layers.append(block(self.in_channels,out_channels,stride))
    #         self.in_channels = out_channels * block.expansion
    #     return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

# def resnet18():
#     return ResNet(BasicBlock,[2,2,2,2])
