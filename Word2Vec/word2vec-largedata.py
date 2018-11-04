#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
word2vec
largedata
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import collections, math, os, random, zipfile
from sklearn.manifold import TSNE
from six.moves import urllib, xrange


def download(url, filename, filesize):
    """
    下载文件并交验大小
    网速不好的话，直接下载工具下载后放到downloads文件夹中即可
    """
    filepath = 'downloads/'+filename
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url + filename, filepath)
    statinfo = os.stat(filepath)
    if statinfo.st_size == filesize:
        print('Found and verified', filepath)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filepath + '. Can you get to it with a browser?')
    return filepath


def read_data(filename):
    """
    读取压缩文件中的内容
    """
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dictionary(words,vocabulary_size):
    """
    计算词频，去掉低频率单词
    返回字典
    """
    # 利用UNK代表低频率单词
    # 统计其余vocabulary_size-1个单词的频率
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    # 为单词做唯一编号
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # 单词列表转编号
    # 如果单词在字典中，获取其编号
    # 不再字典中，则设置为0
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)

    # 设置UNK的记数
    count[0][1] = unk_count

    # 字典KV互换，改为(index, word)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    """
    创建训练集
    """
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    global data_index

    # create empty batch ndarray using 'batch_size'
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    # for each element in our calculated span, append the datum at 'data_index' and increment 'data_index' moduli the amount of data
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # loop for 'batch_size' // 'num_skips'
    for i in range(batch_size // num_skips):
         # target label at the center of the buffer
        target = skip_window
        targets_to_avoid = [skip_window]
        # loop for 'num_skips'
        for j in range(num_skips):
            # loop through all 'targets_to_avoid'
            while target in targets_to_avoid:
                # pick a random index as target
                target = random.randint(0, span - 1)
            # put it in 'targets_to_avoid'
            targets_to_avoid.append(target)
            # set the skip window in the minibatch data
            batch[i * num_skips + j] = buffer[skip_window]
            # set the target in the minibatch labels
            labels[i * num_skips + j, 0] = buffer[target]
        # add the data at the current 'data_index' to the buffer
        buffer.append(data[data_index])
        # increment 'data_index'
        data_index = (data_index + 1) % len(data)
    # return the minibatch data and corresponding labels
    return batch, labels


def train(vocabulary_size,reverse_dictionary):
    """
    训练模型
    """
    batch_size = 128 #训练集大小
    embedding_size = 128  # dimension of the embedding vector
    skip_window = 1  # how many words to consider to left and right
    num_skips = 2  # how many times to reuse an input to generate a label
    valid_size = 16  # size of random set of words to evaluate similarity on
    valid_window = 100  # only pick development samples from the first 'valid_window' words
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64  # number of negative examples to sample

    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # 初始化系数，查找embeddings
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # NCE loss计算公式，并作为梯度下降法优化函数输入
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
            labels=train_labels, inputs=embed, num_sampled=num_sampled,
            num_classes=vocabulary_size))
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # similarity计算公式
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings/norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        # 初始化
        init = tf.global_variables_initializer()

    num_steps = 100001
    steplist = []
    losslist = []
    with tf.Session(graph=graph) as session:
        init.run()
        average_loss = 0
        for step in xrange(num_steps):
            # 生成测试数据,并训练
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            # 2000步骤输出一次Average loss
            if step % 2000 == 0:
                steplist.append(step)
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step ", step, ": ", average_loss)
                losslist.append(average_loss)
                average_loss = 0

            # 10000步计算一次similarity
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    # 类似于knn
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)

        final_embeddings = normalized_embeddings.eval()
    return final_embeddings,valid_examples,sim


def plot_with_labels(low_dim_embs, labels):
    """
    可视化输出
    """
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    # plt.savefig(output/tsne.png)
    plt.show()


if __name__ == '__main__':
    # 加载数据
    url = 'http://mattmahoney.net/dc/'
    filename = 'text8.zip'
    filesize = 31344016
    afile = download(url, filename, filesize)
    words = read_data(afile)
    print('data size:', len(words))

    # 创建字典去掉低频率单词
    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = build_dictionary(words,vocabulary_size)

    # 释放内存
    del words
    print('most common words (+UNK):', count[:10])
    print('sample data:', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # 创建一个训练集
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    # 训练
    final_embeddings,valid_examples,sim=train(vocabulary_size,reverse_dictionary)

    # t-distributed stochastic neighbor embedding进行降维
    # 并进行可视化
    try:
        plot_only = 500
        # create the t-SNE object with 2 components, PCA initialization, and 5000 iterations
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        # fit the TSNE dimensionality reduction technique to the word vector embedding
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels)
    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

    # 进行校验
    valid_size = 16
    for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        # 类似于knn
        top_k = 8
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "nearest to %s:" % valid_word
        for k in xrange(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = "%s %s," % (log_str, close_word)
        print(log_str)
