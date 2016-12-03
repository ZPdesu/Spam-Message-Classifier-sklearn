# -*- coding: utf-8 -*-
import numpy as np
import sklearn
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
from sklearn import svm
from sklearn import metrics
import json
from scipy import sparse, io
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
import re
import load_data
from word_vector import TfidfVectorizer
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

class Predictor:
    def __init__(self, test_data, test_target):
        self.test_data = test_data
        self.test_target = test_target

    def sample_predict(self, clf):
        test_result = clf.predict(self.test_data)
        print 'ZhuPei is very very cool'
        print metrics.classification_report(self.test_data, test_result)
        print metrics.confusion_matrix(self.test_data, test_result)

    def new_predict(self, clf):
        test_result = clf.predict(self.test_data)
        with open('result/predict_label.txt', 'wt') as f:
            for i in range(len(test_result)):
                f.writelines(test_result[i])
        self.test_target = test_result
        print 'write over'


