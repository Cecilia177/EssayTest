from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from feature import extract_features
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X, y = extract_features()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)

# print("X_train shape:", X_train.shape)
# print("Y_train shape:", len(y_train))

clf = svm.SVC(gamma=5, C=10)  # kernel = "rbf"

scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
print(scores)

# clf.fit(X_train, y_train)
# print(clf.predict(X_test))
# score = clf.score(X_test, y_test)
# print(score)


train_size, train_loss, test_loss = learning_curve(
    clf, X, y, cv=3, scoring='accuracy',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
)
train_loss_mean = np.mean(train_loss, axis=1)
test_loss_mean = np.mean(test_loss, axis=1)

plt.figure()
plt.plot(train_size, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(train_size, test_loss_mean, 'o-', color='g', label='Cross-validation')
# plt.legend('best')
plt.show()




