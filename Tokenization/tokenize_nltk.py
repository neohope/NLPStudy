#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
nltk分词
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def tokenization(text):
    """
    分词
    """
    tokens=word_tokenize(text)
    return tokens


def lemmatization(text):
    """
    词形还原，复数变单数
    """
    tokens=word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def stemming(text):
    """
    词干，时态变为正常
    """
    tokens = word_tokenize(text.lower())
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return tokens


def stop_words():
    """
    停顿词
    """
    stopwords = nltk.corpus.stopwords.words('english')
    return stopwords


def remove_stopwords(tokens):
    newtokens = []
    for token in tokens:
        if token not in stop_words():
            newtokens.append(token)
    return newtokens


def normalization(doc,stopwords):
    """
    标准化处理
    分词
    词形还原
    去除停顿词
    """
    normalized_doc=normalize_text(tweet)
    tokens = word_tokenize(normalized_tweet)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    newtokens = []
    for token in tokens:
        if token not in stopwords:
            newtokens.append(token)

    return normalized_doc,newtokens


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


def normalize_docs(docs):
    """
    向量化
    """
    lemmatizer = WordNetLemmatizer()

    normalized_docs = []
    for document in docs:
        text = normalize_text(document)
        nl_text = ''
        for word in word_tokenize(text):
            if word not in stop_words():
                nl_text += (lemmatizer.lemmatize(word)) + ' '
        normalized_docs.append(nl_text)
    return normalized_docs


if __name__ == "__main__":
    text = "I love programming and programming also loves me"

    # 分词
    tokens=tokenization(text)
    print(tokens)

    # 词形还原并分词
    tokens=lemmatization(text)
    print(tokens)

    # 词干并分词
    tokens=stemming(text)
    print(tokens)

    # 停顿词
    newtokens=remove_stopwords(tokens)
    print(newtokens)

    # 标准化处理
    tweet = 'China has new "AI Development Plan." I think this will really help Chinese and global AI. http://www.gov.cn/zhengce/content/2017-07/20/content_5211996.htm'
    normalized_tweet,newtokens = normalization(tweet,stop_words())
    print(normalized_tweet)
    print(set(newtokens))
