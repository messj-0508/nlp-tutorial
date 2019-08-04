# code by Tae Hwan Jung @graykode
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

'''
tf.reset_default_graph函数用于清除默认图形堆栈并重置全局默认图形. 
默认图形是当前线程的一个属性.该tf.reset_default_graph函数只适用于当前线程.
当一个tf.Session或者tf.InteractiveSession激活时调用这个函数会导致未定义的行为.
调用此函数后使用任何以前创建的tf.Operation或tf.Tensor对象将导致未定义的行为.
'''
tf.reset_default_graph()

sentences = [ "i like dog", "i love coffee", "i hate milk"]

'''
以下四步是制作词典
'''
word_list = " ".join(sentences).split() # word segmentation
word_list = list(set(word_list)) # remove repetitive elements
word_dict = {w: i for i, w in enumerate(word_list)} # pairs-word:index
number_dict = {i: w for i, w in enumerate(word_list)} # pairs-index: word
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # number of steps ['i like', 'i love', 'i hate']
n_hidden = 2 # number of hidden units

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
3.设置网络结构：输入-全连接层（tanh）-输出层（softmax）
4.设置loss（交叉熵）和优化器
5.训练（5000轮）
6.预测
7.测试
'''
X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, number of steps, number of Vocabulary]
Y = tf.placeholder(tf.float32, [None, n_class])

input = tf.reshape(X, shape=[-1, n_step * n_class]) # [batch_size, n_step * n_class]
H = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
d = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

tanh = tf.nn.tanh(d + tf.matmul(input, H)) # [batch_size, n_hidden]
model = tf.matmul(tanh, U) + b # [batch_size, n_class]

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction =tf.argmax(model, 1)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = make_batch(sentences)

for epoch in range(10000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
predict =  sess.run([prediction], feed_dict={X: input_batch})

# Test
input = [sen.split()[:2] for sen in sentences]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n] for n in predict[0]])
