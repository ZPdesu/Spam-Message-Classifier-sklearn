# -*- coding: utf-8 -*-

import json
from scipy import sparse, io
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from bayes_Trainer import Trainer_bayes
from bayes_Predictor import Predictor


def split_data(content, label):
    training_data, test_data, training_target, test_target = train_test_split(
        content, label, test_size=0.1, random_state=0)
    return training_data, test_data, training_target, test_target


class Evaluator:
    clf = joblib.load('model/bayes_estimator.pkl')

    def __init__(self, training_data, training_target, test_data, test_target):
        self.trainer = Trainer_bayes(training_data, training_target)
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
    evaluator = Evaluator(training_data.todense(), training_target, test_data.todense(), test_target)
    evaluator.train()
    #evaluator.cross_validation()
    evaluator.predict(type='sample_data')