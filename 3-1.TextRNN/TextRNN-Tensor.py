'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
# 在该网络的收敛速度上，tensor 略胜一筹 。
import tensorflow as tf
import numpy as np

# 重置默认图
tf.reset_default_graph()

sentences = [ "i like dog", "i love coffee", "i hate milk"]

'''
制作数据和词典
'''
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)

# TextRNN Parameter
'''
设置超参数：
1. 输入的长度
2. 每个cell包含的神经单元
'''
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
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch

# Model
'''
建模分以下步骤：
1.设置输入输出的占位符
2.随机正态分布初始化网络参数
3.设置网络结构：输入-RNN-输出层（softmax）
4.设置loss（交叉熵）和优化器（Adam）
5.训练（5000轮）
6.预测
'''

## 1
X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, n_step, n_class]
Y = tf.placeholder(tf.float32, [None, n_class])         # [batch_size, n_class]

## 2
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))


## 3
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# outputs : [batch_size, n_step, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2]) # [n_step, batch_size, n_hidden]
outputs = outputs[-1] # [batch_size, n_hidden]
model = tf.matmul(outputs, W) + b # model : [batch_size, n_class]

## 4
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

prediction = tf.cast(tf.argmax(model, 1), tf.int32)

## 5
# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = make_batch(sentences)

for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
 
## 6       
input = [sen.split()[:2] for sen in sentences]

predict =  sess.run([prediction], feed_dict={X: input_batch})
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n] for n in predict[0]])