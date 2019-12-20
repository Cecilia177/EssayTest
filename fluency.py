import re
from sentence import Sentence


def corpus_prep(corpus_path):
    """
    Preprocess the corpus.
    Return:
        A set of part-of-speech 5~grams of corpus.

    """

    corpus_speech = set()
    pattern = re.compile("/\w+\s*")
    with open(corpus_path, 'r') as f:
        sent = f.readline()
        while sent:
            text = "".join(re.split(pattern, sent))
            text = text.replace('[', '')
            text = text.replace(']', '')
            print(text)
            ss = Sentence(text=text, language="ch")
            ss.get_part_of_speech()
            for s in ss.speeches_gram.keys():
                corpus_speech.add(s)
            sent = f.readline()
    return corpus_speech


def get_fluency_score(ref, sentence):
    """
    Evaluate the format similarity between ref and sentence.
    Parameter:
        ref: Class Sentence
        sentence: Class Sentence
    """
    ref.get_part_of_speech()
    sentence.get_part_of_speech()
    speech_grams = sentence.speeches_gram
    # print(speech_grams)
    total_count = 0
    match_count = 0
    for g in speech_grams.keys():
        total_count += speech_grams[g]
        if g in ref.speeches_gram.keys():
            match_count += speech_grams[g]
    return float(match_count) / total_count


if __name__ == '__main__':
    s0 = "我们不必一定去学习如何做到心理健康，这种能力植根于我们自身，就像我们的身体知道如何愈合伤口，如何修复断骨。"
    s1 = "我们没有学习过,怎样变得健康,它建立在我们自己的身体知道,怎样修复一报坏死的骨头或者治愈伤口,以同样的方式."
    s2 = "我们不必学习如何使心理健康,因为它在身体内部塑造我们。这正如我们的身体知道怎样愈合伤口或修复骨折一样."
    s3 = "我们不需要学会怎样来保持精神健康，这是与生俱来的,好像我们的身体知道如何痊愈一个伤口或修复一个受伤的骨头"
    corpus_speech = corpus_prep("M:\\DATA\\People's_Daily_2014\\1ref.txt")
    ss0 = Sentence(text=s0, language="ch")
    ss1 = Sentence(text=s1, language="ch")
    ss2 = Sentence(text=s2, language="ch")
    ss3 = Sentence(text=s3, language="ch")
    print(get_fluency_score(ss0, ss0))
    print(get_fluency_score(ss0, ss1))
    print(get_fluency_score(ss0, ss2))
    print(get_fluency_score(ss0, ss3))