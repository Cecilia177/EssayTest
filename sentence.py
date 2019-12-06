import math
import string
from zhon.hanzi import punctuation
import re
import jieba
import jieba.posseg as psg


class Sentence:
    def __init__(self, text="", ngram={}, length=0, flag="ch"):
        self.text = text
        self.ngram = ngram
        self.length = length
        self.flag = flag

    def get_ngram(self):
        """
        Get statistic on n-grams and the number of words of a text
        :param text: String
        :return: An array of the 1~4grams strings and the number of words(segments when Chinese) of the text.
        """
        words_list = [i for i in self.text.split(" ") if i]
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
                if gramkey not in ngram.keys():
                    ngram[gramkey] = 1
                else:
                    ngram[gramkey] += 1
        self.ngram = ngram

    def remove_puc(self):
        """
        Remove all punctuations.
        :param text: the original text
        :return: the "clean" text
        """
        for c in string.punctuation:
            self.text = self.text.replace(c, "")
        self.text = re.sub("[{}]+".format(punctuation), "", self.text)

    def segment(self):
        """
        Get segmentation of this paragraph.
        :return: String of segmentation aligned with " "
        """
        if self.flag == "ch":
            segments = jieba.cut(self.text)
            self.text = " ".join(segments)
            self.length = len(self.text.split(" "))
        else:
            self.length = len(self.text.split(" "))

    def part_of_speech(self):
        speeches = psg.cut(self.text)
        for word, flag in speeches:
            print(word, ": ", flag)
        return speeches


