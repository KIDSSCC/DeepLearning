from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def findFiles(path):
    """
    根据指定的匹配模式来匹配所有的文件，
    :param path: 匹配模式
    :return: 文件列表
    """
    return glob.glob(path)

# 所有的ascii字符加上四个符号
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    """
    将Unicode格式转换为ASCII格式
    :param s: Unicode格式的字符
    :return:
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

category_lines = {}
all_categories = []
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# 构建全部的语言和名字的映射
for filename in findFiles('./data/names/*.txt'):
    # 前缀名
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    # 每一个文件中都保存了哪些名字，以ASCII进行存储
    category_lines[category] = lines
n_categories = len(all_categories)


def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# 进行训练集和数据集的划分
train_data=dict()
test_data=dict()
for key,value in category_lines.items():
    data = value
    split_ratio = 0.8  # 训练集所占比例
    split_index = int(len(data) * split_ratio)
    train_data[key] = data[:split_index]
    test_data[key] = data[split_index:]

from rnn import RNN
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    # 初始全零的隐藏向量
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        # 对于name中的每一个字母
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()

def evaluate():
    correct=0
    all=0
    for key,value in test_data.items():
        for name in value:
            line_tensor = lineToTensor(name)
            hidden = rnn.initHidden()
            for i in range(line_tensor.size()[0]):
                output, hidden = rnn(line_tensor[i], hidden)
            # 得到结果向量output
            guess, guess_i = categoryFromOutput(output)
            if guess == key:
                correct+=1
        all+=len(value)
    print('Accuract is {}/{}'.format(correct,all))



print_every = 5000
plot_every = 1000
current_loss = 0
all_losses = []
n_epoch=10
for i in range(n_epoch):
    # 遍历训练集中的所有
    count = 0
    for key,value in train_data.items():
        for name in value:
            category_tensor = torch.tensor([all_categories.index(key)], dtype=torch.long)
            line_tensor = lineToTensor(name)
            output, loss = train(category_tensor, line_tensor)
            current_loss += loss
        count+=1
        print('finish {}/{}'.format(count,len(train_data)))

    # 进行一遍验证
    evaluate()

