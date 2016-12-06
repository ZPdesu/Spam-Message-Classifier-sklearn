from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import json
from scipy import sparse, io
from sklearn.externals import joblib
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import numpy as np
from preprocessing_data import split_data
from preprocessing_data import dimensionality_reduction


class Trainer_bayes:
    def __init__(self, training_data, training_target):
        self.training_data = training_data
        self.training_target = training_target
        self.clf = GaussianNB()


    def train_classifier(self):
        self.clf.fit(self.training_data, self.training_target)
        joblib.dump(self.clf, 'model/bayes_estimator.pkl')
        training_result = self.clf.predict(self.training_data)
        print metrics.classification_report(self.training_target, training_result)

    def cross_validation(self):
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
        scores = cross_val_score(self.clf, self.training_data, self.training_target, cv=cv, scoring='f1_macro')
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




def bayes_train(train_data, train_target):

    model = GaussianNB()
    model.fit(train_data, train_target)
    expected = train_target
    predicted = model.predict(train_data)
    # summarize the fit of the model
    print metrics.classification_report(expected, predicted)
    print metrics.confusion_matrix(expected, predicted)


if '__main__' == __name__:
    content = io.mmread('../Data/word_vector.mtx')
    with open('../Data/train_label.json', 'r') as f:
        label = json.load(f)
    content = content
    training_data, test_data, training_target, test_target = split_data(content, label)
    print np.shape(training_data)
    training_data, test_data = dimensionality_reduction(training_data.todense(), test_data.todense(), type='pca')
    print np.shape(training_data)

    Trainer = Trainer_bayes(training_data.todense(), training_target)
    Trainer.train_classifier()
    #Trainer.cross_validation()

