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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from SVM_Trainer import TrainerLinear
from SVM_Trainer import TrainerRbf
from SVM_Predictor import Predictor
from sklearn import preprocessing


def split_data(content, label):
    training_data, test_data, training_target, test_target = train_test_split(
        content, label, test_size=0.1, random_state=0)
    return training_data, test_data, training_target, test_target


def standardized_data(content, label):
    training_data, test_data, training_target, test_target = split_data(content, label)
    scalar = preprocessing.StandardScaler().fit(training_data)
    training_data_transformed = scalar.transform(training_data)
    test_data_transformed = scalar.transform(test_data)
    return training_data_transformed, test_data_transformed, training_target, test_target


class Evaluator:
    clf = joblib.load('model/SVM_linear_estimator.pkl')

    def __init__(self, training_data, training_target, test_data, test_target):
        self.trainer = TrainerLinear(training_data, training_target)
        self.predictor = Predictor(test_data, test_target)

    def train(self):
        #self.trainer.learn_best_param()
        self.trainer.train_classifier()
        joblib.dump(self.clf, 'model/Terminal_estimator.pkl')
        Evaluator.clf = joblib.load('model/Terminal_estimator.pkl')

    def cross_validation(self):
        self.trainer.cross_validation()

    def predict(self, type):
        if type == 'sample_data':
            self.predictor.sample_predict(Evaluator.clf)
        elif type == 'new_data':
            self.predictor.new_predict(Evaluator.clf)


if '__main__' == __name__:
    content = io.mmread('../Data/word_vector.mtx')
    with open('../Data/train_label.json', 'r') as f:
        label = json.load(f)
    training_data, test_data, training_target, test_target = split_data(content, label)
    evaluator = Evaluator(training_data, training_target, test_data, test_target)
    evaluator.train()
    #evaluator.cross_validation()
    #evaluator.predict(type='sample_data')
