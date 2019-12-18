import numpy as np
from sklearn.metrics import mean_squared_error
from sentence import Sentence
from feature import get_bleu_score
from LSA import LSA
from gensim.models import KeyedVectors
from scipy.linalg import norm


word_vectors = KeyedVectors.load("C:\\Users\\Cecilia\\AppData\\Local\\Temp\\vectors.kv")

s0 = "我们不必一定去学习如何做到心理健康"   # reference
s1 = "我们不需要去学习怎样让肌肉健康"   # candidate1

mylsa = LSA([], '')

ss0 = Sentence(flag="ch", text=s0)
ss1 = Sentence(flag="ch", text=s1)
ss0.preprocess()
ss1.preprocess()
doc_matrix = [ss0.pure_text, ss1.pure_text]
for sentence in doc_matrix:
    mylsa.parse(sentence)
mylsa.build_count_matrix()
print(mylsa.A)
mylsa.TFIDF()
print(mylsa.A)
# mylsa.svd_cal()


def get_sentence_vec(s, tf):
    print(tf)
    vec = np.zeros(64)
    words_list0 = s.pure_text.split()
    for index, w in enumerate(words_list0):
        if w in word_vectors.vocab:
            vec += tf[index] * word_vectors[w]
    return vec


vec0 = get_sentence_vec(ss0, mylsa.A[:, 0])
vec1 = get_sentence_vec(ss1, mylsa.A[:, 1])
sim = np.dot(vec0, vec0) / (norm(vec0) * norm(vec1)) if (norm(vec0) * norm(vec1)) != 0 else 0
print(vec0)
print(vec1)
print("similarity:", sim)


