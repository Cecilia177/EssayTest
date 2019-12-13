import numpy as np
from sklearn import svm
from feature import extract_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import learning_curve, validation_curve, train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from correlation import pearson_cor
from sklearn.metrics import make_scorer


def cross_val(X_train, y_train, score):
    parameters = [
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
        {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.01, 1, 5, 10, 100]},
        # {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [3, 4, 5], 'gamma': [0.01, 1, 5, 10, 100]}
    ]
    svc = svm.SVC(probability=False)
    clf = GridSearchCV(estimator=svc, param_grid=parameters, scoring=score, cv=3, return_train_score=True)
    clf.fit(X_train, y_train)
    return clf.best_estimator_, clf.best_params_


def learning_plot(clf, X_train, y_train, score):
    train_size, train_scores, test_scores = learning_curve(
        clf, X_train, y_train, cv=3, scoring=score,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
    )
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    print(test_score_mean)
    plot(train_size, [train_score_mean, test_score_mean])


def validation_plot(clf, X_train, y_train, score):
    gamma_range = [0.1, 1, 10, 25, 50]
    C_range = [1, 10, 50, 100, 1000]
    # validate parameter gamma
    train_scores_g, test_scores_g = validation_curve(
        clf, X_train, y_train, param_name='gamma', param_range=gamma_range, cv=3, scoring=score)
    # validate parameter C
    train_scores_c, test_scores_c = validation_curve(
        clf, X_train, y_train, param_name='C', param_range=C_range, cv=3, scoring=score
    )
    # get average scores of training and tests
    train_scores_g_mean = np.mean(train_scores_g, axis=1)
    test_scores_g_mean = np.mean(test_scores_g, axis=1)
    scores_list_g = [train_scores_g_mean, test_scores_g_mean]
    plot(gamma_range, scores_list_g)

    train_scores_c_mean = np.mean(train_scores_c, axis=1)
    test_scores_c_mean = np.mean(test_scores_c, axis=1)
    scores_list_c = [train_scores_c_mean, test_scores_c_mean]
    plot(C_range, scores_list_c)


def plot(x_value, y_value_list):
    plt.figure()
    plt.plot(x_value, y_value_list[0], 'o-', color='r', label='Training')
    plt.plot(x_value, y_value_list[1], 'o-', color='g', label='Cross-validation')
    # plt.legend('best')
    plt.show()


# get features and tags for classification task
X_cf, y_cf = extract_features(1)
# regularize the features data
min_max_scaler = MinMaxScaler()
X_cf_minmax = min_max_scaler.fit_transform(X_cf)

# define a scoring function
score_func = make_scorer(pearson_cor, greater_is_better=True)

# get the best classifier through CV
best_clf, best_params = cross_val(X_cf_minmax, y_cf, score_func)
print("best clf:", best_clf)
print("best para:", best_params)

learning_plot(best_clf, X_cf_minmax, y_cf, score_func)
# validation_plot(best_clf, X_cf_minmax, y_cf)



