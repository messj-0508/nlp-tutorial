'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# 设置网络参数属性的别名
dtype = torch.FloatTensor

'''
制作数据和词典，该数据集是单词级的
'''
sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
)

print(sentence)

word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}

# TextLSTM Parameters
'''
设置超参数：
1. 输入的长度
2. 每个cell包含的神经单元数
'''
n_class = len(word_dict)
max_len = len(sentence.split())
n_hidden = 5

'''
制作数据集：单词 → index编号 → onehot向量,预测下个词
'''
def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))

# Model
'''
建模：
1.定义TextRNN类，包括网络参数，和前向传播：输入→Bi-LSTM→输出层（softmax）
2.设置loss（交叉熵）和优化器(Adam)
3.训练（10000轮）
4.测试
'''

## 1
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Parameter(torch.randn([n_hidden * 2, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        hidden_state = Variable(torch.zeros(1*2, len(X), n_hidden))   # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, len(X), n_hidden))     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]
        return model

input_batch, target_batch = make_batch(sentence)

#print(input_batch, target_batch)

model = BiLSTM()

## 2
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## 3
# Training
for epoch in range(10000):
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

## 4
predict = model(input_batch).data.max(1, keepdim=True)[1]
print(sentence)
print([number_dict[n.item()] for n in predict.squeeze()])