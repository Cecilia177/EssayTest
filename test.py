import re
from zhon.hanzi import punctuation
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

pattern = re.compile("\\d+\\.")
# pattern2 = re.compile("^\\d+\\.")
str1 = "46. _我" \
       "们不需要学习怎样像精神变得健康;,我们的身体知道怎样愈 合伤或者修复断裂的骨头，而精神就同身体的这种试一样 建拍着我们"
str2 = "46..ni889.89"
str3 = "dss46."
ignored_chars = [" ", "_", "-"]


def match(str):
    if re.match(pattern, str):
        print(str, ": ok")
    else:
        print(str, ": fail")


def splittt(str):
    for char in ignored_chars:
        str.replace(char, "")  # remove useless blank spaces
    str.replace(";", "")
    print("text: ", str)
    lists = re.split(pattern, str, maxsplit=1)
    text = "".join(lists) if lists[0] == "" else str
    print("text: ", text)


corpus = [
    '我们 不必 学习 如何 变得 心灵 健康',
    '我们 不必 一定 去 学习 如何 做到 心理健康'
]

words = CountVectorizer.fit_transform(corpus)
tfidf = TfidfTransformer.fit_transform(words)
print(CountVectorizer.get_feature_names())
print(tfidf)
