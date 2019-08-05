'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  Reference : https://github.com/ioatr/textcnn
'''
import tensorflow as tf
import numpy as np

# 重置默认图
tf.reset_default_graph()

# Text-CNN Parameter
'''
设置超参数：
1. embedding维度
2. 句子长度
3. 类别数（0-1分类）
4. 一维卷积核的size
5. 每种size对应的卷积核的个数
'''
embedding_size = 2 # n-gram
sequence_length = 3
num_classes = 2 # 0 or 1
filter_sizes = [2,2,2] # n-gram window
num_filters = 3

# 3 words sentences (=sequence_length is 3)
'''
制作数据和词典
'''
sentences = ["i love you","he loves me", "she likes baseball", "i hate you","sorry for that", "this is awful"]
labels = [1,1,1,0,0,0] # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)

'''
制作对应的数据集:one-hot
'''
inputs = []
for sen in sentences:
    inputs.append(np.asarray([word_dict[n] for n in sen.split()]))

outputs = []
for out in labels:
    outputs.append(np.eye(num_classes)[out]) # ONE-HOT : To using Tensor Softmax Loss function

# Model
'''
建模：
1.设置输入输出的占位符
2.随机正态分布初始化Embedding矩阵，并将输入转为词向量，添加“channel”维度
3.设置网络,可以参考： https://www.cnblogs.com/bymo/p/9675654.html
4.训练（5000轮）
5.测试
'''

## 1
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, num_classes])

## 2
W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedded_chars = tf.nn.embedding_lookup(W, X) # [batch_size, sequence_length, embedding_size]
embedded_chars = tf.expand_dims(embedded_chars, -1) # add channel(=1) [batch_size, sequence_length, embedding_size, 1]

## 3
# 公共模型（前半部分）
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]))

    conv = tf.nn.conv2d(embedded_chars, # [batch_size, sequence_length, embedding_size, 1]
                        W,              # [filter_size(n-gram window), embedding_size, 1, num_filters(=3)]
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    h = tf.nn.relu(tf.nn.bias_add(conv, b))
    pooled = tf.nn.max_pool(h,
                            ksize=[1, sequence_length - filter_size + 1, 1, 1], # [batch_size, filter_height, filter_width, channel]
                            strides=[1, 1, 1, 1],
                            padding='VALID')
    pooled_outputs.append(pooled) # dim of pooled : [batch_size(=6), output_height(=1), output_width(=1), channel(=1)]

num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, num_filters) # h_pool : [batch_size(=6), output_height(=1), output_width(=1), channel(=1) * 3]
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # [batch_size, ]

# 训练模型（后半部分）
# Model-Training
Weight = tf.get_variable('W', shape=[num_filters_total, num_classes], 
                    initializer=tf.contrib.layers.xavier_initializer())
Bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
model = tf.nn.xw_plus_b(h_pool_flat, Weight, Bias)  
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 预测模型（后半部分）
# Model-Predict
hypothesis = tf.nn.softmax(model)
predictions = tf.argmax(hypothesis, 1)


## 4
# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: outputs})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%06d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

## 5
# Test
test_text = 'sorry hate you'
tests = []
tests.append(np.asarray([word_dict[n] for n in test_text.split()]))

predict = sess.run([predictions], feed_dict={X: tests})
result = predict[0][0]
if result == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")