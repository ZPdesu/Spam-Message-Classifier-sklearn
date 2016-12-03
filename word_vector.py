# -*- coding: utf-8 -*-
import numpy as np
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
import json
import re
from scipy import sparse, io
import load_data


# 将连续的数字转变为长度的维度
def process_cont_numbers(content):
    digits_features = np.zeros((len(content), 16))
    for i, line in enumerate(content):
        for digits in re.findall(r'\d+', line):
            length = len(digits)
            if 0 < length <= 15:
                digits_features[i, length-1] += 1
            elif length > 15:
                digits_features[i, 15] += 1
    return process_cont_numbers


# 正常分词，非TFID
class MessageCountVectorizer(sklearn.feature_extraction.text.CountVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer


# 用TFID生成对应词向量
class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        #analyzer = super(TfidfVectorizer, self).build_analyzer()
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer


# 生成词向量并进行存储
def vector_word():
    with open('RawData/train_content.json', 'r') as f:
        content = json.load(f)
    with open('RawData/train_label.json', 'r') as f:
        label = json.load(f)
    '''
        vec_count = MessageCountVectorizer(min_df=2, max_df=0.8)
        data_count = vec_count.fit_transform(content)
        name_count_feature = vec_count.get_feature_names()
    '''

    vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8)
    data_tfidf = vec_tfidf.fit_transform(content)
    name_tfidf_feature = vec_tfidf.get_feature_names()

    io.mmwrite('Data/word_vector.mtx', data_tfidf)

    with open('Data/train_label.json', 'w') as f:
        json.dump(label, f)
    with open('Data/vector_type.json', 'w') as f:
        json.dump(name_tfidf_feature, f)

if '__main__' == __name__:
    vector_word()
    print ' OK '
