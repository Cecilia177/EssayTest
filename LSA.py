import numpy as np
from numpy import zeros
from scipy.linalg import svd
import math


class LSA(object):
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords       # exclude words without actual meaning like "the", "and", etc.
        self.ignorechars = ignorechars   # exclude punctuation marks
        self.wdict = {}
        self.dcount = 0  # number of docs
        self.keys = []
        self.A = 0
        self.U = 0
        self.s = 0
        self.Vh = 0

    def parse(self, doc):
        """
        Parse the document, aka. calculate the words in a doc. wdict["book"] = ["3", "4"] represents the word "book"
        appears in both 3th and 4th doc
        :param doc: String of doc
        :return: none
        """
        # words = [x for x in doc.split("/") if x]
        words = doc.split()
        for w in words:
            # lowercase all letters and delete the ignored words
            b = bytearray(w.lower(), 'utf-8')
            w = (b.translate(None, delete=bytearray(self.ignorechars, 'utf-8'))).decode('utf-8')

            if w in self.stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1

    def build_count_matrix(self):
        """
        Build the T*D matrix. The element Axy is the times Term x appears in Doc y.
        :return: None
        """
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) >= 1]
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])  # initialize the matrix
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1

    def printA(self):
        print(self.A)

    def TFIDF(self):
        """
        Modify the counts with TF-IDF
        :return:
        """
        # sum up numbers on the same column, how many keys every doc contains
        WordsPerDoc = np.sum(self.A, axis=0)
        # sum up numbers on the same row, how many docs the key belongs to
        DocsPerWord = np.sum(np.asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i, j] = (float(self.A[i, j]) / WordsPerDoc[j]) * math.log(float(cols) / DocsPerWord[i])

    def svd_cal(self):
        """
        Singular Value Decomposition
        :return: Matrix U, tuple s, Matrix Vh
        """
        self.U, self.s, self.Vh = svd(self.A, full_matrices=False)

    def get_similarity(self, k, i, j):

        """
        Choose top K values from tuple s and calculate the similarity between doc i and doc j.

        :return: cosine similarity value
        """

        sigma = np.zeros([k, k])
        for x in range(k):
            sigma[x, x] = self.s[x]

        a = np.mat(sigma.dot(self.Vh[:, i][0: k]))
        b = np.mat(sigma.dot(self.Vh[:, j][0: k]))
        num = float(a * b.T)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = num / denom
        return cos


titles = [
    "我们 不必 学习 如何 变得 心灵 健康 这 就 跟 我们 身体 知道 如何 愈合 一道 小伤 或是 治疗 断骨 一样 自然天成",
    "我们 不必 一定 去 学习 如何 做到 心理健康 这种 能力 植根于 我们 自身 就 像 我们 的 身体 知道 如何 愈合 伤口 如何 修复 断骨",
]
stopwords = []
ignorechars = ''

mylsa = LSA(stopwords, ignorechars)
for title in titles:
    mylsa.parse(title)
mylsa.build_count_matrix()
mylsa.printA()
print("")
mylsa.TFIDF()
mylsa.printA()
mylsa.svd_cal()
print(mylsa.get_similarity(2, 0, 1))

