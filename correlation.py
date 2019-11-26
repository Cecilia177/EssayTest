import numpy as np
import pandas as pd


def pearson(a, b):
    """
    Calculate the Pearson correlation coefficient between a and b
    :param a:
    :param b: both a and b are Array and their lengths are the same.
    :return: Pearson correlation coefficient value.
    """
    x1 = pd.Series(a)
    y1 = pd.Series(b)
    return x1.corr(y1, method='pearson')


def spearman(a, b):
    """
    The same like the above function pearson except that it's Spearman correlation coefficient value here.
    """
    x1 = pd.Series(a)
    y1 = pd.Series(b)
    return x1.corr(y1, method='spearman')


a = [1, 2, 0.8]
b = [4, 5, 6]
print(pearson(a, b))
print(spearman(a, b))
