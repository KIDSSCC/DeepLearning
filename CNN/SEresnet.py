import torch
import torch.nn as nn
import torch.nn.functional as F



class SEBlock(nn.Module):
    def __init__(self,in_channels,out_channel,expansion,stride,r,downsample=False):
        super(SEBlock, self).__init__()
        self.expansion=expansion
        # 残差结构
        self.residual=nn.Sequential(
            nn.Conv2d(in_channels,out_channel,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel*self.expansion,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel*self.expansion),
            nn.ReLU(inplace=True)
        )
        # 短路连接，或下采样
        self.shortcut=nn.Sequential()
        if downsample:
            self.shortcut.add_module("conv",nn.Conv2d(in_channels,out_channel*self.expansion,kernel_size=1,stride=stride))
            self.shortcut.add_module("BN",nn.BatchNorm2d(out_channel*self.expansion))

        # Squeeze-and-Excitation
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channel * self.expansion, out_channel * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel * self.expansion // r, out_channel * self.expansion),
            nn.Sigmoid()
        )

    def forward(self,input):
        output=self.residual(input)
        squeeze = self.squeeze(output)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(output.size(0), output.size(1), 1, 1)
        output = output * excitation.expand_as(output) + self.shortcut(input)
        return F.relu(output)

class SEresnet(nn.Module):
    def __init__(self):
        super(SEresnet, self).__init__()
        block=[2,2,2,2]
        self.in_channels = 64
        self.expansion=1
        self.c1=nn.Sequential(
            nn.Conv2d(3,self.in_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # c2阶段：
        self.conv2 = nn.Sequential(
            SEBlock(64, 64, self.expansion, 1,r=16),
            SEBlock(64 * self.expansion, 64, self.expansion, 1,r=16)
        )
        # c3阶段：
        self.conv3 = nn.Sequential(
            SEBlock(64, 128, self.expansion, 2, 16,True),
            SEBlock(128 * self.expansion, 128, self.expansion, 1,r=16),
        )
        # c4阶段：
        self.conv4 = nn.Sequential(
            SEBlock(128, 256, self.expansion, 2,16, True),
            SEBlock(256 * self.expansion, 256, self.expansion, 1,r=16),
        )
        # c5阶段
        self.conv5 = nn.Sequential(
            SEBlock(256, 512, self.expansion, 2, 16,True),
            SEBlock(512 * self.expansion, 512, self.expansion, 1,r=16),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, 10)

    def forward(self,input):
        output=self.c1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
# class BasicResidualSEBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride, r=16):
#         super().__init__()
#
#         self.residual = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
#             nn.BatchNorm2d(out_channels * self.expansion),
#             nn.ReLU(inplace=True)
#         )
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels * self.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
#                 nn.BatchNorm2d(out_channels * self.expansion)
#             )
#
#         # SE部分
#         self.squeeze = nn.AdaptiveAvgPool2d(1)
#         self.excitation = nn.Sequential(
#             nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         shortcut = self.shortcut(x)
#         residual = self.residual(x)
#
#         squeeze = self.squeeze(residual)
#         squeeze = squeeze.view(squeeze.size(0), -1)
#         excitation = self.excitation(squeeze)
#         excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
#
#         x = residual * excitation.expand_as(residual) + shortcut
#
#         return F.relu(x)


# class SEResNet(nn.Module):
#     def __init__(self, block, block_num, class_num=100):
#         super().__init__()
#
#         self.in_channels = 64
#
#         self.pre = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#
#         self.stage1 = self._make_stage(block, block_num[0], 64, 1)
#         self.stage2 = self._make_stage(block, block_num[1], 128, 2)
#         self.stage3 = self._make_stage(block, block_num[2], 256, 2)
#         self.stage4 = self._make_stage(block, block_num[3], 512, 2)
#
#         self.linear = nn.Linear(self.in_channels, class_num)
#
#     def forward(self, x):
#         x = self.pre(x)
#
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#
#         x = F.adaptive_avg_pool2d(x, 1)
#         x = x.view(x.size(0), -1)
#
#         x = self.linear(x)
#
#         return x
#
#     def _make_stage(self, block, num, out_channels, stride):
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels * block.expansion
#
#         while num - 1:
#             layers.append(block(self.in_channels, out_channels, 1))
#             num -= 1
#
#         return nn.Sequential(*layers)


if __name__=='__main__':
    input = torch.ones(4, 3, 32, 32)
    net = SEresnet()
    print(net)
    output = net(input)