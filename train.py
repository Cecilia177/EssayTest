from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


class Classifier(object):
    def __init__(self):
        self.clf = ""

    def train(self, x, y):
        self.clf = svm.SVC(decision_function_shape='ovo', kernel='poly')
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)




