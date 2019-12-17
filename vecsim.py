import gensim
import jieba
import numpy as np
from scipy.linalg import norm
from sentence import Sentence
import jieba.posseg as psg
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors


def vector_similarity(s1, s2, vecs, stopwords):

    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(64)
        valid_words = []
        for word in words:
            if word not in stopwords and word in vecs.vocab:
                v += vecs[word]
                valid_words.append(word)
        return v / len(valid_words) if len(valid_words) > 0 else v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2)) if (norm(v1) * norm(v2)) != 0 else 0


if __name__ == '__main__':
    # model_path = "H:\\Download\\news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
    # print("-----start loading model---------")
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    # word_vectors = model.wv
    # fname = get_tmpfile("vectors.kv")
    # word_vectors.save(fname)
    word_vectors = KeyedVectors.load("C:\\Users\\Cecilia\\AppData\\Local\\Temp\\vectors.kv")
    with open('C:\\Users\\Cecilia\\Desktop\\stopwords.txt', 'r+') as f:
        stopwords = f.read().split("\n")

    s0 = "我们不必一定去学习如何做到心理健康这种能力植根于我们自身就像我们的身体知道如何愈合伤口如何修复断骨"

    s1 = "我们没有学习过怎样变得健康它建立在我们自己的身体知道怎样修复一报坏死的骨头或者治愈伤口以同样的方式"   # 0.5'
    s2 = "我们不必去学习怎样变得心理健康心理健康和我们的身体知道如何治疗一个伤口或者使一根断骨愈合的这种方式一样是我们内在天生的."  # 1.5'
    print(vector_similarity(s0, s1, word_vectors, []))
    print(vector_similarity(s0, s2, word_vectors, []))



