from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
dataset_path = "M:\\DATA\\20newsgroups\\20news-bydate-train"

# returned dataset is a scikit-learn bunch
twenty_train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
twenty_train_local = load_files(dataset_path, categories=categories, shuffle=True, random_state=42)

# target_names are the requested categories, which are the names of sub-folders
print(twenty_train_local.target_names)
# data are the files
print(len(twenty_train.filenames))

# target is the array storing the category ids
print(twenty_train_local.target[:10])


X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo', kernel='poly')
clf.fit(X, Y)
print(clf.predict([[4], [8]]))

