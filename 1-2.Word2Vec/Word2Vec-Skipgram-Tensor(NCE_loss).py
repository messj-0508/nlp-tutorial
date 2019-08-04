'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  reference : https://github.com/golbin/TensorFlow-Tutorials/blob/master/04%20-%20Neural%20Network%20Basic/03%20-%20Word2Vec.py
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 重置默认图
tf.reset_default_graph()

# 3 Words Sentence
sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]


word_sequence = " ".join(sentences).split()

'''
制作词典
'''
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}

'''
设置超参数：
1. 批训练样例数
2. embedding维度
3. 负采样数目
4. 词典大小
'''
# Word2Vec Parameter
batch_size = 20
embedding_size = 2 # To show 2 dim embedding graph
num_sampled = 10 # for negative sampling, less than batch_size
voc_size = len(word_list)

'''
随机批训练数据集制作
'''
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    # np.random.choice可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回。
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels

# Make skip gram of one size window
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

    for w in context:
        skip_grams.append([target, w])


# Model
'''
建模分以下步骤：
1.设置输入输出的占位符
2.随机正态分布初始化Embedding矩阵，并将输入转为词向量
3.设置网络参数
4.设置loss（交叉熵）和优化器
5.训练（5000轮）
6.画图展示
'''

## 1
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) # To use tf.nn.nce_loss, [batch_size, 1]

## 2
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

## 3 tf.nn.nce_loss是word2vec的skip-gram模型的负例采样方式的函数 参看https://www.cnblogs.com/xiaojieshisilang/p/9284634.html
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

## 4 Loss and optimizer
cost = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

## 5 Training
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(5000):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
        _, loss = sess.run([optimizer, cost], feed_dict={inputs: batch_inputs, labels: batch_labels})

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    # tf.Tensor.eval() 功能：当默认的会话被指定之后可以通过其获取一个张量的取值。 
    trained_embeddings = embeddings.eval()

## 6
for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()