# code by Tae Hwan Jung @graykode
# 在该网络的收敛速度上，torch 略胜一筹 。
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

'''
以下四步是制作词典
'''
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # n-1 in paper
n_hidden = 2 # h in paper
m = 2 # m in paper

'''
制作数据集：单词 → index编号 
'''
def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))
    
    '''
    1. 进行word-embedding
    2. 进入连接层（tanh）
    3. 输出（输入是连接层的输出和网络的输入）
    '''
    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m) # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U) # [batch_size, n_class]
        return output

model = NNLM()

'''
设置loss（交叉熵）和优化器(Adam)
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

'''
转换数据类型
'''
input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.LongTensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))

# Training
for epoch in range(5000):
    # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
    optimizer.zero_grad()

    # 前向传播
    output = model(input_batch)
    
    # 以下3步计算loss，显示（可选）
    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    # 反向传播求梯度
    loss.backward()
    # 更新参数
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
