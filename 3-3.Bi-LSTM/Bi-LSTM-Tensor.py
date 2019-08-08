#coding:utf-8
'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import tensorflow as tf
import numpy as np

# 重置默认图
tf.reset_default_graph()


# 该数据集是单词级的
sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
)

# 制作词典
word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}

'''
设置超参数：
1. 词典的大小
2. 句子长度
3. cell中神经元数
'''
n_class = len(word_dict)
n_step = len(sentence.split())
n_hidden = 5

'''
制作数据集：单词 → index编号 → onehot向量
'''
def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[n] for n in words[:(i + 1)]]
        input = input + [0] * (n_step - len(input))
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch

# Model
'''
建模分以下步骤：
1.设置输入输出的占位符
2.设置网络结构：输入→Bi-LSTM→输出层（softmax）
3.设置loss（交叉熵）和优化器（Adam）
4.训练（10000轮）
5.预测
'''

## 1
# Bi-LSTM Model
X = tf.placeholder(tf.float32, [None, n_step, n_class])
Y = tf.placeholder(tf.float32, [None, n_class])

## 2
W = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

# outputs : [batch_size, len_seq, n_hidden], states : [batch_size, n_hidden]
outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, X, dtype=tf.float32)

outputs = tf.concat([outputs[0], outputs[1]], 2) # output[0] : lstm_fw, output[1] : lstm_bw
outputs = tf.transpose(outputs, [1, 0, 2]) # [n_step, batch_size, n_hidden]
outputs = outputs[-1] # [batch_size, n_hidden]

model = tf.matmul(outputs, W) + b

## 3
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

prediction = tf.cast(tf.argmax(model, 1), tf.int32)

## 4
# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = make_batch(sentence)

for epoch in range(10000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

## 5
predict =  sess.run([prediction], feed_dict={X: input_batch})
print(sentence)
print([number_dict[n] for n in [pre for pre in predict[0]]])