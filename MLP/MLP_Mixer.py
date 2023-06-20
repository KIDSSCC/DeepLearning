import torch
from torch import nn
from einops.layers.torch import Rearrange
from torchvision import datasets, transforms
import torch.nn.functional as F


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



class MLP(nn.Module):
    def __init__(self,dim,inner_dim):
        super(MLP, self).__init__()
        drop=0.1
        self.fc=nn.Sequential(
            nn.Linear(dim,inner_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(inner_dim,dim),
            nn.Dropout(drop)
        )
    def forward(self,input):
        output=self.fc(input)
        return output


class Mixer(nn.Module):
    def __init__(self,dim,n_patch,token_dim,channel_dim):
        super(Mixer, self).__init__()
        self.token_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            MLP(n_patch,token_dim),
            Rearrange('b d n -> b n d')
        )
        self.channel_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim,channel_dim)
        )

    def forward(self,input):
        output=input+self.token_mixer(input)
        output=output+self.channel_mixer(output)
        return output

class MLP_Mixer(nn.Module):
    def __init__(self,in_channel,dim,patch_size,depth,token_dim,channel_dim):
        super(MLP_Mixer, self).__init__()
        n_patch=(28//patch_size)**2
        self.prepatch=nn.Sequential(
            nn.Conv2d(in_channel, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])
        for i in range(depth):
            self.mixer_blocks.append(Mixer(dim,n_patch, token_dim, channel_dim))

        self.classifier_1=nn.LayerNorm(dim)
        self.classifier_2=nn.Linear(dim,10)

    def forward(self,input):
        output=self.prepatch(input)
        for layer in self.mixer_blocks:
            output=layer(output)
        output=self.classifier_1(output)
        output= output.mean(dim=1)
        return F.log_softmax(self.classifier_2(output),dim=1)



# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)
#
# class MixerBlock(nn.Module):
#
#     def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
#         super().__init__()
#
#         self.token_mix = nn.Sequential(
#             nn.LayerNorm(dim),
#             Rearrange('b n d -> b d n'),
#             FeedForward(num_patch, token_dim, dropout),
#             Rearrange('b d n -> b n d')
#         )
#
#         self.channel_mix = nn.Sequential(
#             nn.LayerNorm(dim),
#             FeedForward(dim, channel_dim, dropout),
#         )
#
#     def forward(self, x):
#
#         x = x + self.token_mix(x)
#
#         x = x + self.channel_mix(x)
#
#         return x
#
# class MLPMixer(nn.Module):
#     def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
#         super().__init__()
#         self.num_patch = (image_size // patch_size) ** 2
#         self.to_patch_embedding = nn.Sequential(
#             nn.Conv2d(in_channels, dim, patch_size, patch_size),
#             Rearrange('b c h w -> b (h w) c'),
#         )
#
#         self.mixer_blocks = nn.ModuleList([])
#         for _ in range(depth):
#             self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
#
#         self.layer_norm = nn.LayerNorm(dim)
#
#         self.mlp_head = nn.Sequential(
#             nn.Linear(dim, num_classes)
#         )
#
#     def forward(self, x):
#
#         x = self.to_patch_embedding(x)
#
#         for mixer_block in self.mixer_blocks:
#             x = mixer_block(x)
#
#         x = self.layer_norm(x)
#
#         x = x.mean(dim=1)
#
#         return self.mlp_head(x)

dim=10
model = MLP_Mixer(1,100,7,4,400,400).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
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