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



# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
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



# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
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


train_data=dict()
test_data=dict()

for key,value in category_lines.items():
    n_samples = len(value)
    split_index = int(n_samples * 0.8)
    random.shuffle(value)
    train_data[key] = value[:split_index]
    test_data[key] = value[split_index:]


# Find letter index from all_letters, e.g. "a" = 0
# 确定字母索引
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
# 单个字母转换为独热向量
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor
#整个单词转换为张量
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

from LSTM_rewrite import mylstm
from LSTM import LSTM
from rnn import RNN
hidden_size=128
lstm=LSTM(57,128,18)
print(lstm)

# 根据输出的张量确定所属的语言
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    # 从列表l中随机选择一个元素返回
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(dataset):
    # 随机选择一种语言
    category = randomChoice(all_categories)
    # 从这种语言对应的名字中随机选择一个名字
    line = randomChoice(dataset[category])
    # 结果标签向量？
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    # 返回了当前的语言，当前的名字，以及语言和名字各自对应的张量
    return category, line, category_tensor, line_tensor


criterion = nn.NLLLoss()
learning_rate = 0.05
def train(category_tensor, line_tensor):
    # 初始全零的隐藏向量
    hidden=lstm.init_hidden()
    lstm.zero_grad()

    for i in range(line_tensor.size()[0]):
        # 对于name中的每一个字母
        output,hidden= lstm(line_tensor[i],hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in lstm.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

n_iters = 200000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []
all_accuracy=[]

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(train_data)
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

        correct = 0
        all = 0
        for key, value in test_data.items():
            for name in value:
                line_tensor = lineToTensor(name)
                hidden = lstm.init_hidden()
                for i in range(line_tensor.size()[0]):
                    output, hidden = lstm(line_tensor[i], hidden)
                # 得到结果向量output
                guess, guess_i = categoryFromOutput(output)
                if guess == key:
                    correct += 1
            all += len(value)
        all_accuracy.append(correct/all)

plt.figure()
plt.plot(all_losses)
plt.title('Loss change')

plt.figure()
plt.plot(all_accuracy)
plt.title('Accuracy change')


correct=0
all=0
for key,value in test_data.items():
    for name in value:
        line_tensor = lineToTensor(name)
        hidden = lstm.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden= lstm(line_tensor[i], hidden)
        # 得到结果向量output
        guess, guess_i = categoryFromOutput(output)
        if guess == key:
            correct+=1
    all+=len(value)
print('Accuract is {}/{}'.format(correct,all))

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000
def evaluate(line_tensor):
    hidden = lstm.init_hidden()

    for i in range(line_tensor.size()[0]):
        output ,hidden= lstm(line_tensor[i],hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample(category_lines)
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.title("Predict Matrix")
plt.show()