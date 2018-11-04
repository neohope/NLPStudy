#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
向量化
只考虑了词频，没考虑顺序，有区别于word2vec，后者考了了顺序
"""

import collections
import numpy as np
import pandas as pd
from Tokenization import tokenize_nltk
from sklearn.feature_extraction.text import CountVectorizer


def document_vectorization(normalized_docs):
    """
    向量化
    """

    # 通过计算词率，将单词按出现频率排序
    words=" ".join(normalized_docs).split()
    count= collections.Counter(words).most_common()
    features=[c[0] for c in count]
    #print(features)

    # 统计每个句子的词率
    training_examples = []
    for doc in normalized_docs:
        doc_feature_values = np.zeros(len(features))
        for word in tokenize_nltk.word_tokenize(doc):
            if word in features:
                index = features.index(word)
                doc_feature_values[index] += 1
        training_examples.append(doc_feature_values)
    #print(training_examples)

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(normalized_docs)
    freqs = [(word, matrix.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
    sorted_freqs = sorted(freqs, key=lambda x: -x[1])

    feature_names = vectorizer.get_feature_names()
    corpus_index = [n for n in normalized_docs]
    df = pd.DataFrame(matrix.todense(), index=corpus_index, columns=feature_names)
    return sorted_freqs,df


if __name__ == "__main__":
    docs = [' I love programming', 'Programming also loves me']
    normalized_docs = tokenize_nltk.normalize_docs(docs)
    print(normalized_docs)

    sorted_freqs, df = document_vectorization(normalized_docs)
    print(sorted_freqs)
    print(df)

