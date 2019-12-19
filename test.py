import numpy as np
from sklearn.metrics import mean_squared_error
from sentence import Sentence
from feature import get_bleu_score
from LSA import LSA
from gensim.models import KeyedVectors
from scipy.linalg import norm


A = np.array([[0, 1, 2], [2, 3, 4]])
b1 = np.sum(A, axis=1)

print(b1)
print(np.asarray(A > 0, 'i'))
# word_vectors = KeyedVectors.load("C:\\Users\\Cecilia\\AppData\\Local\\Temp\\vectors.kv")
# vec0 = get_sentence_vec(ss0, mylsa.A[:, 0], word_vectors)
# vec1 = get_sentence_vec(ss1, mylsa.A[:, 1], word_vectors)
# sim = np.dot(vec0, vec1) / (norm(vec0) * norm(vec1)) if (norm(vec0) * norm(vec1)) != 0 else 0
# print(vec0)
# print(vec1)
# print("similarity:", sim)


