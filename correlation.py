import numpy as np


def pearson(a, b):
    """
    Calculate the Pearson correlation coefficient between a and b
    :param a:
    :param b: both a and b are Array and their lengths are the same.
    :return: Pearson correlation coefficient value.
    """

    return np.corrcoef(a, b)[1][1]

def spearman(a, b):
    """

    :param a:
    :param b:
    :return:
    """

