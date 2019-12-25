from sklearn.linear_model import LinearRegression
from feature import extract_data
from train import learning_plot
from sklearn.model_selection import ShuffleSplit, learning_curve, train_test_split
from learningcurve import plot_learning_curve
import matplotlib.pyplot as plt
import numpy as np
from train import plot
from correlation import pearson_cor
from sklearn.metrics import make_scorer

feature_list, score_list = extract_data()
model = LinearRegression()
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

# title_rg = r"Learning Curves (SVR, rbf kernel)"
# plot_learning_curve(model, title_rg, feature_list, score_list, ylim=(0.0, 0.8),
#                     cv=3, n_jobs=4, scoring=None)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(feature_list, score_list, test_size=0.33, random_state=42)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(y_predict)
print(pearson_cor(y_predict, y_test))

score_func = make_scorer(pearson_cor, greater_is_better=True)
train_size, train_scores, test_scores = learning_curve(
        model, feature_list, score_list, cv=3, scoring=score_func,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
    )
print(train_scores)
print(test_scores)
train_score_mean = np.mean(train_scores, axis=1)
test_score_mean = np.mean(test_scores, axis=1)
plot(train_size, [train_score_mean, test_score_mean])
