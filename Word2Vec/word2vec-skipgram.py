#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
word2vec
skip gram
"""

import numpy as np
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import collections


def load_data(filename):
    """
    加载数据文件
    标准化处理
    """
    # 加载数据文件
    with open(filename) as fi:
        sentences = fi.readlines()

    # 标准化处理
    normalized_sentences = []
    for sentence in sentences:
        normalized_sentences.append(normalize_text(sentence))
    return normalized_sentences


def normalize_text(text):
    """
    标准化处理
    转换为小写，并去除网址、数字以及特殊符号
    """
    text = text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))', '', text)
    text = re.sub('@[^\s]+', '', text)
    text = re.sub('#([^\s]+)', '', text)
    text = re.sub('[:;>?<=*+()&,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]', ' ', text)
    text = re.sub('[\d]', '', text)
    text = text.replace(".", '')
    text = text.replace("'", '')
    text = text.replace("`", '')
    text = text.replace("'s", '')
    text = text.replace("/", ' ')
    text = text.replace("\"", ' ')
    text = text.replace("\\", '')
    # text =  re.sub(r"\b[a-z]\b", "", text)
    text = re.sub('\s+', ' ', text).strip()
    return text


def create_dict(normalized_sentences):
    """
    创建字典及索引字典
    """
    # 计算词频
    words = " ".join(normalized_sentences).split()
    count = collections.Counter(words).most_common()
    print("Word count top 5 ", count[:5])

    # 构建字典
    unique_words = [i[0] for i in count]
    dic = {w: i for i, w in enumerate(unique_words)}
    voc_size = len(dic)
    print(voc_size)

    # 创建词的索引，就是词的排序
    idx = [dic[word] for word in words]
    print('Sample data first 10 ', idx[:10], words[:10])

    return words,idx,unique_words,voc_size


def create_skipgram_pairs(words,idx):
    """
    创建skip_gram_pairs
    """
    # 构建数据，对每个词，都做
    # [[data[i-1],data[i+1]],data[i]]
    # [[words[i-1],words[i+1]],words[i]]
    cbow_pairs = []
    for i in range(1, len(idx) - 1):
        cbow_pairs.append([[idx[i - 1], idx[i + 1]], idx[i]]);
    print('Context pairs rank ids', cbow_pairs[:5])

    cbow_pairs_words = []
    for i in range(1, len(words) - 1):
        cbow_pairs_words.append([[words[i - 1], words[i + 1]], words[i]]);
    print('Context pairs words', cbow_pairs_words[:5])

    # 创建skip-gram pairs
    # 其实就是把上面的三元组，拆成两个二元组
    # (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
    skip_gram_pairs_words = []
    for c in cbow_pairs_words:
        skip_gram_pairs_words.append([c[1], c[0][0]])
        skip_gram_pairs_words.append([c[1], c[0][1]])
    print('skip-gram pairs words', skip_gram_pairs_words[:5])

    skip_gram_pairs = []
    for c in cbow_pairs:
        skip_gram_pairs.append([c[1], c[0][0]])
        skip_gram_pairs.append([c[1], c[0][1]])
    print('skip-gram pairs', skip_gram_pairs[:5])

    return skip_gram_pairs_words, skip_gram_pairs


def get_random_skipgram_pairs(size,skipgram_pairs):
    """
    随机抽取skip_gram_pairs
    """
    assert size < len(skip_gram_pairs)
    rdm = np.random.choice(range(len(skip_gram_pairs)), size, replace=False)

    X = []
    Y = []
    for r in rdm:
        X.append(skipgram_pairs[r][0])
        Y.append([skipgram_pairs[r][1]])
    return X, Y


def train(batch_size,embedding_size,num_sampled,voc_size):
    """
    训练模型
    """
    X = tf.placeholder(tf.int32, shape=[batch_size])  # inputs
    Y = tf.placeholder(tf.int32, shape=[batch_size, 1])  # labels

    with tf.device("/cpu:0"):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, X)

    # 创建NCE Loss计算函数，并作为Adam优化函数输入
    nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
    nce_biases = tf.Variable(tf.zeros([voc_size]))
    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, Y, embed, num_sampled, voc_size))
    optimizer = tf.train.AdamOptimizer(1e-1).minimize(loss)

    # tf进行训练
    epochs = 10000
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            batch_inputs, batch_labels = get_random_skipgram_pairs(batch_size, skip_gram_pairs)
            _, loss_val = session.run([optimizer, loss], feed_dict={X: batch_inputs, Y: batch_labels})
            # 每1000次输入一次loss
            if epoch % 1000 == 0:
                print("Loss at ", epoch, loss_val)
        trained_embeddings = embeddings.eval()

    return trained_embeddings


if __name__ == '__main__':
    #加载数据
    #这里的输入我找了生活大爆炸某一集的英文字幕
    filename = "downloads/test.srt"
    normalized_sentences = load_data(filename)

    #创建字典
    words,idx,unique_words,voc_size=create_dict(normalized_sentences)

    #创建skip_gram_pairs
    skip_gram_pairs_words, skip_gram_pairs=create_skipgram_pairs(words,idx)

    # 测试随机抽取skip_gram_pairs函数
    print('testint get_random_skipgram_pairs ', get_random_skipgram_pairs(3,skip_gram_pairs))

    # 训练
    batch_size = 20
    embedding_size = 2
    num_sampled = 15
    trained_embeddings=train(batch_size,embedding_size,num_sampled,voc_size)

    # 可视化展示
    # Show top 30 words
    if trained_embeddings.shape[1] == 2:
        labels = unique_words[:30]
        for i, label in enumerate(labels):
            x, y = trained_embeddings[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2),textcoords='offset points', ha='right', va='bottom')
        #plt.savefig("output/word2vec_00_30.png")
        plt.show()

    # Show 30-60
    if trained_embeddings.shape[1] == 2:
        labels = unique_words[30:60]
        for i, label in enumerate(labels):
            x, y = trained_embeddings[i,:]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2),textcoords='offset points', ha='right', va='bottom')
        #plt.savefig("output/word2vec_30_60.png")
        plt.show()
