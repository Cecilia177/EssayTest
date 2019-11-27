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

        S = np.mat(self.s[0: k])
        a = np.mat(S.dot(self.Vh[:, i][0: k]))
        b = np.mat(S.dot(self.Vh[:, j][0: k]))
        num = float(a * b.T)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = num / denom
        return cos

# stopwords = ["的", "是"]
# ignorewords = [",", ".", "。", " "]


titles = [
    "The Neatest Little Guide to Stock Market Investing",
    "Investing For Dummies, 4th Edition",
    "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
    "The Little Book of Value Investing",
    "Value Investing: From Graham to Buffett and Beyond",
    "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
    "Investing in Real Estate, 5th Edition",
    "Stock Investing For Dummies",
    "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
]
stopwords = ['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to']
ignorechars = ''',:'!'''


mylsa = LSA(stopwords, ignorechars)
docs = []
path = "C:\\Users\\Cecilia\\Desktop\\"
filename = ["ref2.txt", "ref2.txt"]
# for file in filename:
#     with open(path + file, 'r') as f:
#         docs.append(f.read())
#         f.close()
for title in titles:
    mylsa.parse(title)
mylsa.build_count_matrix()
mylsa.printA()
print("")
mylsa.TFIDF()
mylsa.printA()
mylsa.svd_cal()
print(mylsa.get_similarity(3, 1, 2))

