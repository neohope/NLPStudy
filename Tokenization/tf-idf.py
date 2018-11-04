#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
词频
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Tokenization import tokenize_nltk


def tf_idf(docs):
    """
    Term frequency and inverse term frequency
    """
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0)
    matrix = vectorizer.fit_transform(docs)

    freqs = [(word, matrix.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
    tf = sorted(freqs, key=lambda x: -x[1])

    feature_names = vectorizer.get_feature_names()
    corpus_index = [n for n in docs]
    df = pd.DataFrame(matrix.todense(), index=corpus_index, columns=feature_names)
    return tf,df


if __name__ == "__main__":
    docs = [' I love programming', 'Programming also loves me']
    normalized_docs = tokenize_nltk.normalize_docs(docs)

    tf,tdidf=tf_idf(normalized_docs)
    print(tf)
    print(df)