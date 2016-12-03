# -*- coding: utf-8 -*-
from numpy import *
import json


# 加载原始数据，进行分割
def load_message():
    content = []
    label = []

    with open('RawData/junk_message.txt') as fr:
        lines = fr.readlines()
        num = len(lines)
        for i in range(num):
            message = lines[i].split('\t')
            label.append(message[0])
            content.append(message[1])
    return num, content, label


# 将分割后的原始数据存到json
def data_storage(content, label):
    with open('RawData/train_content.json', 'w') as f:
        json.dump(content, f)
    with open('RawData/train_label.json', 'w') as f:
        json.dump(label, f)


if '__main__' == __name__:
    num, content, label = load_message()
    data_storage(content, label)
