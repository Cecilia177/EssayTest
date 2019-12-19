import gensim
import jieba
import numpy as np
from scipy.linalg import norm
from sentence import Sentence
import jieba.posseg as psg
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import math


def vector_similarity(id1, id2, vecs, stopwords, tf_idf, keys):
    """
    Calculate two sentences
    Parameters:
        s1, s2: class Sentence, with pure_text composed of phrases segmented by blank spaces.
        vecs: word_vectors file.
        stopwords: a list containing stopwords to ignore when getting the sentence vector.
        tf1, tf2: tf-idf lists of s1 and s2, with the same shape.
    Return:
        A float, as the similarity of Sentence s1 and s2.
    """

    def sentence_vector_with_tf(keys, tf):
        v = np.zeros(64)
        for i, t in enumerate(tf):
            if t != 0:
                word = keys[i]
                print(word, ":", t)
                if word not in stopwords and word in vecs.vocab:
                    v += t * vecs[word]
        return v

    tf1 = tf_idf[:, id1]
    tf2 = tf_idf[:, id2]
    v1, v2 = sentence_vector_with_tf(keys=keys, tf=tf1), sentence_vector_with_tf(keys=keys, tf=tf2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2)) if (norm(v1) * norm(v2)) != 0 else 0


if __name__ == '__main__':
    # model_path = "H:\\Download\\news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
    # print("-----start loading model---------")
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    # word_vectors = model.wv
    # fname = get_tmpfile("vectors.kv")
    # word_vectors.save(fname)

    # load local word vectors
    word_vectors = KeyedVectors.load("C:\\Users\\Cecilia\\AppData\\Local\\Temp\\vectors.kv")
    # with open('C:\\Users\\Cecilia\\Desktop\\stopwords.txt', 'r+') as f:
    #     stopwords = f.read().split("\n")

    s0 = "我们不必一定去学习如何做到心理健康这种能力植根于我们自身就像我们的身体知道如何愈合伤口如何修复断骨"
    s1 = "我们没有学习过怎样变得健康它建立在我们自己的身体知道怎样修复一报坏死的骨头或者治愈伤口以同样的方式"   # 0.5'
    s2 = "我们不必去学习怎样变得心理健康心理健康和我们的身体知道如何治疗一个伤口或者使一根断骨愈合的这种方式一样是我们内在天生的."  # 1.5'

    tfidf = np.loadtxt("C:\\Users\\Cecilia\\Desktop\\tfidf.txt")
    keys = np.loadtxt("C:\\Users\\Cecilia\\Desktop\\keys.txt", dtype=str, delimiter='/n')
    print(vector_similarity(id1=0, id2=8, vecs=word_vectors, stopwords=[], tf_idf=tfidf, keys=keys))



    # print(vector_similarity(0, 1, word_vectors, stopwords=[], tf_idf=tfidf, keys=keys))
    # print(tfidf)
    # print(docs)



