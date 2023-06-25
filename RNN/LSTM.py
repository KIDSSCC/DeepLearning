import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(LSTM, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=1
        # self.hidden=(torch.zeros(1, 128), torch.zeros(1, 128))

        self.lstm=nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.linear=nn.Linear(self.hidden_size,self.output_size)

    def forward(self,input,hidden):
        output,hidden=self.lstm(input,hidden)
        output=self.linear(output)
        # print(output)
        output=F.relu(output)
        output=F.log_softmax(output,dim=1)
        # print(output)
        return output,hidden

    def init_hidden(self):
        return (torch.zeros(1, 128), torch.zeros(1, 128))


if __name__=='__main__':
    input=torch.ones(1,57)
    hidden=(torch.zeros(3, 128), torch.zeros(3, 128))
    net=LSTM(57,128,18)
    output,hidden=net(input,hidden)
    print(output.size())
