from sklearn.datasets import make_moons, make_circles, make_classification
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

classifier = SVC(gamma=2, C=1)
count = 1
for ds_cnt, ds in enumerate(datasets):
    print("This is NO.", count, "training.")
    X, y = ds
    # print("X size:", X.shape)
    print("X:", X)
    # X = StandardScaler().fit_transform(X)
    print("transformed X:", X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    classifier.fit(X_train, y_train)
    print(classifier.predict(X_test))
    score = classifier.score(X_test, y_test)
    print(score)
    count += 1
