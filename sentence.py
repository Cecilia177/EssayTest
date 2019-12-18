import math
import string
from zhon.hanzi import punctuation
import re
import jieba
import jieba.posseg as psg


class Sentence:
    """
    Attributes:
        text -- The original text of the sentence. Updated in method self.remove_punc()
        flag -- "ch" or "en", defines the language.
        seg_length -- number of phrases after segmentation.  Updated in method self.segment()
        pure_text -- made of phrases and blank spaces after removing punctuations. Updated in method self.segment()
        ngram -- A dict, the key of which is 1~4gram string and the value is the count. Updated in method self.get_ngram().
    """
    def __init__(self, flag, text=""):
        self.text = text
        self.ngram = {}
        self.seg_length = 0
        self.flag = flag
        self.pure_text = ""

    def remove_punc(self):
        """
        Replace all punctuations with "sss" in self.text.

        """
        for c in string.punctuation:
            self.text = self.text.replace(c, "sss")
        self.text = re.sub("[{}]+".format(punctuation), "sss", self.text)

    def segment(self):
        """
        Get segmentation of this paragraph.
        Set self.pure_text and self.seg_length
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
        self.pure_text = pure_text            # pure_text is made of phrases and blank spaces
        self.seg_length = length              # seg_length is the number of phrases instead of string length

    def get_ngram(self):
        """
        Set self.ngram

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

    def part_of_speech(self):
        speeches = psg.cut(self.text)
        # for word, flag in speeches:
        #     print(word, ": ", flag)
        return speeches

    def preprocess(self):
        self.remove_punc()
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
