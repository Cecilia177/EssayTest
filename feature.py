import pymysql
from sentence import Sentence
from LSA import LSA
import traceback
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


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
    get_ref_cur = conn.cursor()
    get_detection_cur = conn.cursor()
    get_feature_cur = conn.cursor()
    insert_feature_cur = conn.cursor()

    get_ref_sql = "SELECT courseid, questionid, ref FROM standards"
    try:
        get_ref_cur.execute(get_ref_sql)
        course = get_ref_cur.fetchall()
    except Exception as e:
        print("Error getting courseid and questionid!", e)
    else:
        for courseid, questionid, ref in course:
            reference = Sentence(text=ref)
            print("current question:", questionid, "of", courseid)
            reference.remove_puc()
            reference.segment()
            reference.get_ngram()

            get_detection_sql = "SELECT textid, text FROM detection WHERE courseid = %s and questionid = %s"
            get_feature_sql = "SELECT * FROM features WHERE textid = %s"

            try:
                if get_detection_cur.execute(get_detection_sql, (courseid, questionid)):
                    textid, text = get_detection_cur.fetchone()
                else:
                    print("No quesion", questionid, "of", courseid, "in DETECTION DB.")
                    continue
                current_answer = Sentence(text=text)
                while text:
                    # If the features of current answer is already in DB, than ignore it.
                    get_feature_cur.execute(get_feature_sql, textid)
                    if get_feature_cur.fetchone() is None:
                        current_answer.remove_puc()
                        # Get length ratio
                        lengthratio = format(float(len(current_answer.text)) / len(reference.text), '.2f')
                        current_answer.segment()
                        current_answer.get_ngram()
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

                        insert_feature_sql = "INSERT INTO features(textid, 1gram, 2gram, 3gram, 4gram, lengthratio, lsagrade)" \
                               "VALUES(%s, %s, %s, %s, %s, %s, %s)"
                        try:
                            print("textid:", textid)
                            print("1~4gram:", bleu)
                            print("lengthratio:", lengthratio)
                            print("lsa:", lsagrade)
                            insert_feature_cur.execute(insert_feature_sql, (textid, bleu[0], bleu[1], bleu[2], bleu[3], lengthratio, lsagrade))
                            conn.commit()
                            print("success inserting features!")
                        except Exception as e:
                            print("Error inserting features..", traceback.print_exc())
                            break
                    else:
                        print("Features of textid", textid, "is already in db.")

                    next_record = get_detection_cur.fetchone()
                    textid, text = next_record if next_record is not None else (-1, None)
                    current_answer.text = text
            except Exception as e:
                print("Error getting text...", traceback.print_exc())
    finally:
        get_feature_cur.close()
        get_detection_cur.close()
        get_ref_cur.close()
        insert_feature_cur.close()
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


def extract_features():
    conn = pymysql.connect(host="127.0.0.1",
                           database='essaydata',
                           port=3306,
                           user='root',
                           password='',
                           charset='utf8')
    cur = conn.cursor()
    sql = "SELECT TEXTID, 1GRAM, 2GRAM, 3GRAM, 4GRAM, LENGTHRATIO, LSAGRADE FROM features"
    sql2 = "SELECT Z1 FROM SCORES, DETECTION WHERE TEXTID = %s AND SCORES.STUDENTID = DETECTION.STUDENTID"
    try:
        cur.execute(sql)
        data = cur.fetchall()
        data_list = []
        grade_list = []
        for d in data:
            data_list.append(list(d[1:]))
            cur.execute(sql2, (d[0]))
            grade_list.append(str(cur.fetchone()[0]))
    # except Exception as e:
    #     print("Error getting features!", e)
    finally:
        cur.close()
        conn.close()
    return data_list, grade_list


# get_features()

