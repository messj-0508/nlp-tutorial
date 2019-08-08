'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 设置网络参数属性的别名
dtype = torch.FloatTensor

'''
制作数据和词典，该数据集是字符集的
'''
char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
word_dict = {n: i for i, n in enumerate(char_arr)}
number_dict = {i: w for i, w in enumerate(char_arr)}
n_class = len(word_dict) # number of class(=number of vocab)

seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

# TextLSTM Parameters
'''
设置超参数：
1. 输入的长度
2. 每个cell包含的神经单元数
'''
n_step = 3
n_hidden = 128

'''
制作数据集：字母 → index编号 → onehot向量
'''
def make_batch(seq_data):
    input_batch, target_batch = [], []

    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]] # 'm', 'a' , 'k' is input
        target = word_dict[seq[-1]] # 'e' is target
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))

# Model
'''
建模：
1.定义TextRNN类，包括网络参数，和前向传播：输入-LSTM-输出层（softmax）
2.设置loss（交叉熵）和优化器(Adam)
3.训练（1000轮）
4.测试
'''

## 1
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, X):
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]

        hidden_state = Variable(torch.zeros(1, len(X), n_hidden))   # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1, len(X), n_hidden))     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]
        return model

input_batch, target_batch = make_batch(seq_data)

model = TextLSTM()

## 2
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

output = model(input_batch)

## 3
# Training
for epoch in range(1000):
    optimizer.zero_grad()

    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

## 4
inputs = [sen[:3] for sen in seq_data]

predict = model(input_batch).data.max(1, keepdim=True)[1]
print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])