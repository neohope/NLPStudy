#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
本例实现了一个在固定领域下的Knowledge based Question and answering系统

后端采用jena做语义网络，前端采用命令行交互
前端通过分词及正则表达式，匹配预设问题
匹配到问题模板后，将自然语言问题转换为sparql查询语句，并发给jena查询
然后将查询结果输出
"""

from KBQA import jena_sparql_endpoint
from KBQA import question2sparql


if __name__ == '__main__':
    # TODO 连接Fuseki服务器。
    fuseki = jena_sparql_endpoint.JenaFuseki()
    # TODO 初始化自然语言到SPARQL查询的模块
    q2s = question2sparql.Question2Sparql()

    while True:
        question = input()
        my_query = q2s.get_sparql(question)
        if my_query is not None:
            result = fuseki.get_sparql_result(my_query)
            value = fuseki.get_sparql_result_value(result)

            # TODO 判断结果是否是布尔值，是布尔值则提问类型是"ASK"，回答“是”或者“不知道”。
            if isinstance(value, bool):
                if value is True:
                    print('Yes')
                else:
                    print('I don\'t know. :(')
            else:
                # TODO 查询结果为空，根据OWA，回答“不知道”
                if len(value) == 0:
                    print('I don\'t know. :(')
                elif len(value) == 1:
                    print(value[0])
                    #print(value[0].encode(sys.getfilesystemencoding()))
                else:
                    output = ''
                    for v in value:
                        output += v + u'、'
                    print(output[0:-1])
                    #print(output[0:-1].encode(sys.getfilesystemencoding()))

        else:
            # TODO 自然语言问题无法匹配到已有的正则模板上，回答“无法理解”
            print('I can\'t understand. :(')

        print('#' * 100)
