import math
import string
from zhon.hanzi import punctuation
import re
import jieba
import jieba.posseg as psg


class Sentence:
    def __init__(self, flag, text="", ngram={}, length=0):
        self.text = text
        self.ngram = ngram
        self.length = length
        self.flag = flag
        self.pure_text = ""

    def get_ngram(self):
        """
        Get statistic on n-grams and the number of words of a text
        :param text: String
        :return: An array of the 1~4grams strings and the number of words(segments when Chinese) of the text.
        """
        words_list = [i for i in self.text.split(" ") if i]
        # print("wordslist:", words_list)
        ngram = {}       # key is the actual 1~4gram sequence and value is the number of it
        for i in range(4):
            for j in range(len(words_list)):
                if j + i >= len(words_list):
                    break
                gramkey = words_list[j]
                index = 0
                while index < i:
                    gramkey += " " + words_list[j + index + 1]
                    index += 1
                if gramkey not in ["</s>", "</s> </s>", "</s> </s> </s>"]:
                    if gramkey not in ngram.keys():
                        ngram[gramkey] = 1
                    else:
                        ngram[gramkey] += 1
        self.ngram = ngram

    def remove_puc(self):
        """
        Replace all punctuations with "s".
        :return: the "clean" text
        """
        for c in string.punctuation:
            self.text = self.text.replace(c, "sss")
        self.text = re.sub("[{}]+".format(punctuation), "sss", self.text)

    def segment(self):
        """
        Get segmentation of this paragraph.
        :return: String of segmentation aligned with " "
        """
        sub_sentences = self.text.split("sss")
        new_text = "</s> </s> </s> "
        length = 0
        for s in sub_sentences:
            if s:
                if self.flag == "ch":
                    segments = " ".join(jieba.cut(s))
                    # print(segments)
                else:
                    segments = s
                length += len([i for i in segments.split(" ") if i != ''])
                new_text += segments + " </s> </s> </s> "
        self.text = new_text[:-1]
        pure_text = self.text.replace(" </s> </s> </s>", "")
        pure_text = pure_text.replace("</s> </s> </s> ", "")
        self.pure_text = pure_text   # pure_text is made of phrases and blank spaces
        self.length = length

    def part_of_speech(self):
        speeches = psg.cut(self.text)
        # for word, flag in speeches:
        #     print(word, ": ", flag)
        return speeches

    def preprocess(self):
        self.remove_puc()
        # print("removed punc:", self.text)
        self.segment()
        # print("segmented:", self.text)
        self.get_ngram()


# ss = "我们不必学习如何变得心灵健康,这就跟我们身体知道如何愈合一道小伤或是治疗断骨一样自然天成."
# s = Sentence(text=ss, flag="ch")
# s.preprocess()
# print(s.text)
# print(s.pure_text)
# print(s.pure_text.split())
