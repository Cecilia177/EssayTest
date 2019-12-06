import pymysql
from sentence import Sentence
from LSA import LSA
import traceback
import numpy as np

def get_features():
    pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
    pymysql.converters.conversions = pymysql.converters.encoders.copy()
    pymysql.converters.conversions.update(pymysql.converters.decoders)

    conn = pymysql.connect(host="127.0.0.1",
                           database='essaydata',
                           port=3306,
                           user='root',
                           password='',
                           charset='utf8')
    cur = conn.cursor()
    cur1 = conn.cursor()

    sql = "SELECT courseid, questionid, ref FROM standards"
    try:
        cur.execute(sql)
        course = cur.fetchall()
        print("course: ", course)
    except Exception as e:
        print("Error getting courseid and questionid!", e)
    else:
        for courseid, questionid, ref in course:
            reference = Sentence(text=ref)
            print("answer: ", reference.text)
            reference.remove_puc()
            reference.segment()
            reference.get_ngram()

            sql2 = "SELECT textid, text FROM detection WHERE courseid = %s and questionid = %s"

            try:
                if cur.execute(sql2, (courseid, questionid)):
                    textid, text = cur.fetchone()
                else:
                    continue
                current_answer = Sentence(text=text)
                while text:
                    # print("current_text: ", current_answer.text)
                    current_answer.remove_puc()
                    # Get length ratio
                    # print("reference length: ", reference.text)
                    lengthratio = format(float(len(current_answer.text)) / len(reference.text), '.2f')

                    current_answer.segment()
                    current_answer.get_ngram()
                    # print("current answer length: ", current_answer.length)
                    bleu = get_bleu_score(reference, current_answer)

                    stopwords = []
                    ignorechars = ''
                    mylsa = LSA(stopwords, ignorechars)
                    for sentence in [reference.text, current_answer.text]:
                        mylsa.parse(sentence)
                    mylsa.build_count_matrix()
                    mylsa.TFIDF()
                    mylsa.svd_cal()
                    lsagrade = mylsa.get_similarity(2, 0, 1)

                    sql3 = "INSERT INTO features(textid, 1gram, 2gram, 3gram, 4gram, lengthratio, lsagrade)" \
                           "VALUES(%s, %s, %s, %s, %s, %s, %s)"
                    try:
                        print("textid:", textid)
                        print("1~4gram:", bleu)
                        print("lengthratio:", lengthratio)
                        print("lsa:", lsagrade)

                        cur1.execute(sql3, (textid, bleu[0], bleu[1], bleu[2], bleu[3], lengthratio, lsagrade))
                        conn.commit()

                        print("success inserting features!")
                        textid, text = cur.fetchone()
                        current_answer.text = text
                    except Exception as e:
                        print("Error inserting features..", traceback.print_exc())
                        break
            except Exception as e:
                print("Error getting text...", traceback.print_exc())
    finally:
        cur.close()
        conn.close()


def get_bleu_score(ref, answer):
    ref_ngram = ref.ngram
    answer_ngram = answer.ngram
    total_count = [0] * 4
    match_count = [0] * 4
    for i in range(4):
        total_count[i] += answer.length - i
    for key in answer_ngram.keys():
        if key in ref_ngram.keys():
            # print("bingo: ", key, ngram[key])
            match_count[len(key.split(" ")) - 1] += answer_ngram[key]
    # print("match_count: ", match_count)
    # print("total_count: ", total_count)
    # score = math.exp(sum([math.log(float(a)/b) for a, b in zip(self.match_count, self.total_count)]) * 0.25)
    score_list = []
    for i in range(4):
        score_list.append(format(float(match_count[i] / total_count[i]), ".4f"))
    # score_list.append(score)
    return score_list


get_features()


