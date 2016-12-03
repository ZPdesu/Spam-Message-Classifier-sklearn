from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import json
from scipy import sparse, io


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

    bayes_train(content.todense(), label)