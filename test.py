from sklearn.metrics import make_scorer
from correlation import pearson_cor
from sklearn import svm
import numpy as np


score = make_scorer(pearson_cor, greater_is_better=True)
X = [["1"], ["1"]]
ss = [0, 1]
y = np.array([0, 1])

clf = svm.SVC()
clf = clf.fit(X, y)

print("predict:", clf.predict(X))
print(pearson_cor(clf.predict(X), y))
print(pearson_cor(ss, y))
