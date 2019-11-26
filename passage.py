import math


class Passage:
    class Sentence:
        def __init__(self, content="", sentence_id=0):
            self.content = content
            self.sentence_id = sentence_id

        @staticmethod
        def get_ngram(text):
            """
            Get statistic on n-grams and the number of words of a text
            :param text: String
            :return: An array of the 1~4grams strings and the number of words(segments when Chinese) of the text.
            """

            for i in (' ', '.', ',', '?', '!', "\"", "/"):
                text = text.replace(i, " ")     # only in English text
            words_list = [i for i in text.split(" ") if i]
            words_list_len = len(words_list)
            text = words_list
            ngram = {}       # key is the actual 1~4gram sequence and value is the number of it
            for i in range(4):
                for j in range(len(text)):
                    if j + i >= len(text):
                        break
                    gramkey = text[j]
                    index = 0
                    while index < i:
                        gramkey += " " + text[j + index + 1]
                        index += 1
                    if gramkey not in ngram.keys():
                        ngram[gramkey] = 1
                    else:
                        ngram[gramkey] += 1
            return ngram, words_list_len

    def __init__(self, reference=[], sentences=[], match_count=[0, 0, 0, 0], total_count=[0, 0, 0, 0], bleu_score=0):
        self.reference = reference
        self.sentences = sentences
        self.match_count = match_count
        self.total_count = total_count
        self.bleu_score = bleu_score

    def get_bleu_score(self):
        for index, sentence in enumerate(self.sentences):
            ref_sentence = self.reference[index]
            ngram, words_list_length = self.Sentence.get_ngram(sentence)
            ref_ngram, ref_length = self.Sentence.get_ngram(ref_sentence)

            for i in range(4):
                self.total_count[i] += words_list_length - i
            for key in ngram.keys():
                if key in ref_ngram.keys():
                    # print("bingo: ", key, ngram[key])
                    self.match_count[len(key.split(" ")) - 1] += ngram[key]

        print("match_count: ", self.match_count)
        print("total_count: ", self.total_count)

        # score = math.exp(sum([math.log(float(a)/b) for a, b in zip(self.match_count, self.total_count)]) * 0.25)
        score_list = []
        for i in range(4):
            score_list.append(float(self.match_count[i] / self.total_count[i]))
        # score_list.append(score)
        return score_list


if __name__ == '__main__':
    with open("C:\\Users\\Cecilia\\Desktop\\ref1.txt", 'r') as ref:
        reference = [x for x in ref.read().split("<s>") if x]
        with open("C:\\Users\\Cecilia\\Desktop\\中文1.txt", 'r') as f:
            passage = Passage(reference, [x for x in f.read().split("<s>") if x])
            bleu_scores = passage.get_bleu_score()
            print(bleu_scores)
            f.close()
    ref.close()
