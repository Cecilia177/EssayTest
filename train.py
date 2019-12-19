import numpy as np
from sklearn import svm
from feature import extract_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import learning_curve, validation_curve, train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from correlation import pearson_cor
from sklearn.metrics import make_scorer
from learningcurve import plot_learning_curve
from sklearn.svm import SVR
import pymysql


def cross_val(estimator, params, X_train, y_train, score, cv):

    clf = GridSearchCV(estimator=estimator, param_grid=params, scoring=score, cv=cv, return_train_score=True)
    clf.fit(X_train, y_train)
    return clf.best_estimator_, clf.best_params_


def learning_plot(clf, X_train, y_train, score, cv):
    train_size, train_scores, test_scores, fit_times = learning_curve(
        clf, X_train, y_train, cv=cv, scoring=score,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1], return_times=True
    )
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
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


if __name__ == '__main__':

    conn = pymysql.connect(host="127.0.0.1",
                           database='essaydata',
                           port=3306,
                           user='root',
                           password='',
                           charset='utf8')

    # get features and tags for classification task
    # X_cf, y_cf = extract_data()
    # regularize the features data
    min_max_scaler = MinMaxScaler()
    # X_cf_minmax = min_max_scaler.fit_transform(X_cf)

    parameters = [
            {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
            {'kernel': ['rbf'], 'C': [0.1, 0.5, 1, 10, 100, 1000], 'gamma': [0.01, 1, 5, 10, 100]},
            # {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [3, 4], 'gamma': [0.01, 1, 5, 10, 100]}
        ]
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    # get the best classifier through CV
    # svc = svm.SVC(probability=False)
    # best_clf, best_params = cross_val(svc, params=parameters, X_train=X_cf_minmax,
    #                                   y_train=y_cf, score="accuracy", cv=cv)
    # print("best clf:", best_clf, "best para:", best_params)
    #
    # title = r"Learning Curves (SVM, linear kernel, C=1000)"
    # plot_learning_curve(best_clf, title, X_cf_minmax, y_cf, ylim=(0.3, 1.01),
    #                     cv=cv, n_jobs=4)
    # plt.show()
    # learning_plot(best_clf, X_cf_minmax, y_cf, "accuracy", cv)
    # validation_plot(best_clf, X_cf_minmax, y_cf)

    # define a scoring function
    score_func = make_scorer(pearson_cor, greater_is_better=True)

    # Regression model
    X_rg, y_rg = extract_data(conn)
    print("number of features:", len(X_rg[0]))
    print("number of samples:", len(y_rg))
    X_rg_minmax = min_max_scaler.fit_transform(X_rg)
    svr = SVR()
    # Get the best model through CV
    best_svr, best_params_rg = cross_val(svr, params=parameters, X_train=X_rg_minmax,
                                         y_train=y_rg, score=score_func, cv=cv)
    print("best svr:", best_svr)
    print("best para:", best_params_rg)

    title_rg = r"Learning Curves (SVR, rbf kernel)"
    plot_learning_curve(best_svr, title_rg, X_rg_minmax, y_rg, ylim=(0.0, 0.8),
                        cv=cv, n_jobs=4, scoring=score_func)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X_rg_minmax, y_rg, test_size=0.33, random_state=42)
    best_svr.fit(X_train, y_train)
    y_predict = best_svr.predict(X_test)
    for index, y in enumerate(y_test):
        if abs(y - y_predict[index]) > 0.2:
            print("target:", y, "predict:", y_predict[index])
    print(pearson_cor(y_test, y_predict))
    # print(best_svr.predict([[0, 0, 0, 0, 0, 0]]))
