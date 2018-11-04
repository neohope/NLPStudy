#!usr/bin/python
# encoding=utf-8

"""
jieba分词及词性标定
"""

import jieba.posseg as postag


if __name__ == "__main__":
    doc = "别让别人告诉你你成不了才，即使是我也不行。如果你有梦想的话，就要去捍卫它。那些一事无成的人想告诉你你也成不了大器。如果你有理想的话，就要去努力实现。就这样。"
    words = postag.cut(doc);

    for w in words:
        print(w.word, "/", w.flag, " ", end="")