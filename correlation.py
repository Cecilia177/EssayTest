import pandas as pd


def pearson_cor(y_true, y_predict):
    """
    Calculate the Pearson correlation coefficient between a and b
    :param y_true:
    :param y_predict: both y_true and y_predict are Array and their lengths are the same.
    :return: Pearson correlation coefficient value.
    """
    if type(y_true[0]) != float:
        y_true = [float(s) for s in y_true]
    if type(y_predict[0]) != float:
        y_predict = [float(s) for s in y_predict]
    y1 = pd.Series(y_true)
    y2 = pd.Series(y_predict)
    return y1.corr(y2, method='pearson')


def spearman_cor(y_true, y_predict):
    """
    The same like the above function pearson except that it's Spearman correlation coefficient value here.
    """
    if type(y_true) == str:
        y_true = [float(s) for s in y_true]
    if type(y_predict) == str:
        y_predict = [float(s) for s in y_predict]
    y1 = pd.Series(y_true)
    y2 = pd.Series(y_predict)
    return y1.corr(y2, method='spearman')


# a = ["0.93", "0.1", "1.0"]
# b = [0.8, 0.08, 1.5]
# print(pearson_cor(a, b))
