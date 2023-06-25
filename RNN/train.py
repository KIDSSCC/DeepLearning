import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import *


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)



from LSTM_rewrite import mylstm
hidden_size=128
rnn=mylstm()
criterion = nn.NLLLoss()
learning_rate = 0.05

def train(category_tensor, line_tensor):
    # 初始全零的隐藏向量
    hidden = rnn.init_hidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        # 对于name中的每一个字母
        output,hc= rnn(line_tensor[i],hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def validate():
    correct = 0
    all = 0
    for key, value in category_lines.items():
        for name in value:
            line_tensor = lineToTensor(name)
            hidden = rnn.init_hidden()
            for i in range(line_tensor.size()[0]):
                output, hidden = rnn(line_tensor[i], hidden)
            # 得到结果向量output
            guess, guess_i = categoryFromOutput(output)
            if guess == key:
                correct += 1
        all += len(value)
    print('Accuract is {}/{}'.format(correct, all))

def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output ,hidden= rnn(line_tensor[i],hidden)
    return output


n_iters = 200000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

# begin to train
start = time.time()
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
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
# begin to test
validate()

plt.figure()
plt.plot(all_losses)
plt.savefig('./loss.png')
plt.clf()

# 准备混淆矩阵
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
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
plt.savefig('./matrix.png')
plt.clf()

