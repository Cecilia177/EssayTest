from stanfordcorenlp import StanfordCoreNLP
from sentence import Sentence


# extract the outside NP
def get_phrases(phrase, sentence, model):
    """
    Parameter:
        phrase: 'NP' or 'VP'
        sentence: class Sentence
        model: parsing model
    Return:
        float, average length of phrases.
    """
    parsed_ser = model.parse(sentence.original)
    print(parsed_ser)
    flag = 0
    nps = []
    count = 0
    total_len = 0
    for i in range(len(parsed_ser) - 1):
        if parsed_ser[i: i+2] == phrase and flag == 0:
            flag = 1
            temp = ""
            count += 1
        elif flag == 1:
            if u'\u4e00' <= parsed_ser[i] <= u'\u9fa5':
                temp += parsed_ser[i]
            elif parsed_ser[i] == '(':
                count += 1
            elif parsed_ser[i] == ')':
                count -= 1
                if count == 0:
                    flag = 0
                    nps.append(temp)
    print(nps)
    for n in nps:
        total_len += len(n)
    return float(total_len) / len(nps) if len(nps) != 0 else 0


# if __name__ == '__main__':
#     zh_model = StanfordCoreNLP(r"H:\Download\stanford-corenlp-full-2018-02-27", lang='zh')
#     s0 = "我们不必一定去学习如何做到心理健康，这种能力植根于我们自身，就像我们的身体知道如何愈合伤口，如何修复断骨。"
#     ss0 = Sentence(text=s0, language="ch")
#     np0 = get_phrases(phrase='NP', sentence=ss0, model=zh_model)
#     VP0 = get_phrases(phrase='VP', sentence=ss0, model=zh_model)
#     print(np0)
#     print(VP0)
