# -*- coding: utf-8 -*-
import numpy as np
import sklearn
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
from sklearn import svm
from sklearn import metrics

import json
import re
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


def SVM(data, target):
    svc = svm.SVC(kernel='linear')
    svc.fit(data, target)
    expected = target
    predicted = svc.predict(data)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


if '__main__' == __name__:

    num, content, flag = load_data.load_message()
    with open('RawData/train_content.json', 'r') as f:
        content = json.load(f)

    '''
    vec_count = MessageCountVectorizer(min_df=2, max_df=0.8)
    data_count = vec_count.fit_transform(content)
    name_count_feature = vec_count.get_feature_names()
    '''

    vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8)
    data_tfidf = vec_tfidf.fit_transform(content)
    name_tfidf_feature = vec_tfidf.get_feature_names()

    data_tfidf_test = np.around(data_tfidf,3)

    print data_tfidf_test


    with open('RawData/train_label.json', 'r') as f:
        label = json.load(f)
    SVM(data_tfidf_test, label)
    print type(data_tfidf_test)
    print type(data_tfidf_test[1])
    print type(label)
    print type(label[1])

    #for i in range(num):
        #print name_tfidf_feature[i]





