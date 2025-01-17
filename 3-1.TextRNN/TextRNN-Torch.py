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
制作数据和词典
'''
sentences = [ "i like dog", "i love coffee", "i hate milk"]
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)

'''
设置超参数：
1. 批输入的个数
2. 输入的长度
3. 每个cell包含的神经单元
'''
# TextRNN Parameter
batch_size = len(sentences)
n_step = 2 # number of cells(= number of Step)
n_hidden = 5 # number of hidden units in one cell

'''
制作数据集：单词 → index编号 → onehot向量
'''
def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

# to Torch.Tensor

'''
转换数据类型
'''
input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.Tensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))


'''
建模：
1.定义TextRNN类，包括网络参数，和前向传播：输入-RNN-输出层（softmax）
2.设置loss（交叉熵）和优化器(Adam)
3.训练（5000轮）
4.测试
'''

## 1
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, hidden, X):
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = torch.mm(outputs, self.W) + self.b # model : [batch_size, n_class]
        return model

model = TextRNN()

## 2
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## 3
# Training
for epoch in range(5000):
    optimizer.zero_grad()

    # hidden : [num_layers * num_directions, batch, hidden_size]
    hidden = Variable(torch.zeros(1, batch_size, n_hidden))
    # input_batch : [batch_size, n_step, n_class]
    output = model(hidden, input_batch)

    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

input = [sen.split()[:2] for sen in sentences]

## 4
# Predict
hidden = Variable(torch.zeros(1, batch_size, n_hidden))
predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])