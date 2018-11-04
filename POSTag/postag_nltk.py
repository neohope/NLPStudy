#!usr/bin/python

"""
nltk分词及词性标定
"""

import nltk

if __name__ == "__main__":
    # 测试句子
    doc = "Don’t ever let somebody tell you you can’t do something, not even me. You got a dream, you gotta protect it. People can’t do something themselves, they wanna tell you you can’t do it. If you want something, go get it. Period."

    # 分词
    tokens = nltk.word_tokenize(doc)
    print(tokens)

    # 词性标注
    tagged = nltk.pos_tag(tokens)
    print(tagged)

    # 句法分析
    entities = nltk.chunk.ne_chunk(tagged)
    print(entities)
