import pymysql
from sentence import Sentence
from LSA import LSA
import traceback
import numpy as np
from correlation import pearson_cor
from vecsim import vector_similarity
import gensim
from gensim.models import KeyedVectors

pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
pymysql.converters.conversions = pymysql.converters.encoders.copy()
pymysql.converters.conversions.update(pymysql.converters.decoders)


def get_features():
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
    get_all_detection = conn.cursor()

    get_ref_sql = "SELECT courseid, questionid, ref FROM standards"

    record_count = 0
    try:
        get_ref_cur.execute(get_ref_sql)
        course = get_ref_cur.fetchall()

        word_vectors = KeyedVectors.load("C:\\Users\\Cecilia\\AppData\\Local\\Temp\\vectors.kv")

        for courseid, questionid, ref in course:
            reference = Sentence(text=ref, flag="ch")
            print("current question:", questionid, "of", courseid)
            reference.preprocess()
            doc_matrix = [reference.pure_text]
            get_detection_sql = "SELECT textid, text FROM detection WHERE courseid = %s and questionid = %s"
            get_feature_sql = "SELECT * FROM features WHERE textid = %s"

            try:
                if get_all_detection.execute(get_detection_sql, (courseid, questionid)):
                    detections = get_all_detection.fetchall()
                    # print("all docs:", docs)
                    docs = []
                    textids = []
                    for dt in detections:
                        textids.append(dt[0])
                        docs.append(dt[1])
                        cur_ans = Sentence(text=dt[1], flag="ch")
                        cur_ans.preprocess()
                        doc_matrix.append(cur_ans.pure_text)

                    # build count matrix, tf-idf modification and svd
                    stopwords = []
                    ignorechars = ''
                    mylsa = LSA(stopwords, ignorechars)
                    for doc in doc_matrix:
                        mylsa.parse(doc)
                    mylsa.build_count_matrix()
                    mylsa.TFIDF()
                    mylsa.svd_cal()

                else:
                    print("No quesion", questionid, "of", courseid, "in DETECTION DB.")
                    continue

                i = 1
                for index, text in enumerate(docs):
                    current_answer = Sentence(text=text, flag="ch")
                    textid = textids[index]

                    # If the features of current answer is already in DB, ignore it.
                    # get_feature_cur.execute(get_feature_sql, textids[index])
                    # if get_feature_cur.fetchone() is None:

                    current_answer.preprocess()
                    # Get length ratio
                    lengthratio = format(float(current_answer.seg_length) / reference.seg_length, '.2f')
                    # Get 1~4gram scores
                    bleu = get_bleu_score(reference, current_answer)
                    # Get 1~4gram scores
                    lsagrade = mylsa.get_similarity(10, 0, i)
                    # Get sentence vec similarity

                    doc_tf = mylsa.A[: i]  # the tfidf list of doc i is the NO.i column of mylsa.A


                    with open('C:\\Users\\Cecilia\\Desktop\\stopwords.txt', 'r+') as f:
                        stopwords = f.read().split("\n")
                    answer_words = current_answer.pure_text.replace(" ", "")
                    reference_words = reference.pure_text.replace(" ", "")
                    vec_sim = vector_similarity(answer_words, reference_words, word_vectors, [])

                    insert_feature_sql = "INSERT INTO features(textid, 1gram, 2gram, 3gram, 4gram, lengthratio, lsagrade, vecsim)" \
                           "VALUES(%s, %s, %s, %s, %s, %s, %s, %s)"
                    try:
                        print("Inserting features__textid:", textid, "1~4gram:", bleu, "lengthratio:", lengthratio,
                              "lsa:", lsagrade, "vec:", vec_sim)
                        insert_feature_cur.execute(insert_feature_sql, (textid, bleu[0], bleu[1], bleu[2], bleu[3], lengthratio, lsagrade, vec_sim))
                        conn.commit()
                        print("Success!")
                    except Exception as e:
                        print("Error inserting features..", traceback.print_exc())
                        break
                    i += 1
                record_count += i - 1
            except Exception as e:
                print("Error getting text...", traceback.print_exc())
        print("--------------------Finishing inserting features of", record_count, "text.----------------------")
    except Exception as e:
        print("Error getting courseid and questionid!", e)
    finally:
        get_feature_cur.close()
        get_detection_cur.close()
        get_ref_cur.close()
        insert_feature_cur.close()
        conn.close()


def get_bleu_score(ref, answer):
    """
    paras:
        ref: class Sentence
        answer: class Sentence
    """
    ref_ngram = ref.ngram
    answer_ngram = answer.ngram
    total_count = [0] * 4
    match_count = [0] * 4
    for key in answer_ngram.keys():
        n = len(key.split(" ")) - 1   # key is n-gram
        total_count[n] += 1
        if key in ref_ngram.keys():
            match_count[n] += min(answer_ngram[key], ref_ngram[key])
            print("Got one:", key, "+", min(answer_ngram[key], ref_ngram[key]))
    # score = math.exp(sum([math.log(float(a)/b) for a, b in zip(match_count, total_count)]) * 0.25)
    score_list = []
    for i in range(4):
        score_list.append(format(float(match_count[i] / total_count[i]), ".4f"))
    # score_list.append(score)
    return score_list


def extract_features():
    """
    Get the correlation of score(y) and each feature.
    Return:
        feature_list: the matrix of feature values, shape of which is (M, N).
                --M is the number of samples and N is the number of features(6 for now).
                --features[i][j] is the NO.j feature value of NO.i sample.
            feature sequence is ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade'].
        score_list:
            matrix of scores, shape of which is (M, 1).
                --M is the number of samples.
    """
    conn = pymysql.connect(host="127.0.0.1",
                           database='essaydata',
                           port=3306,
                           user='root',
                           password='',
                           charset='utf8')
    get_all_features_sql = "SELECT textid, 1gram, 2gram, 3gram, 4gram, lengthratio, lsagrade, vecsim FROM features"
    get_questionid_sql = "SELECT questionid FROM detection WHERE textid=%s"
    cur = conn.cursor()
    feature_list = []
    score_list = []
    score_text = {}  # key is textid and value is score(aka. y) of the text
    try:
        cur.execute(get_all_features_sql)
        features = cur.fetchall()
        # Get feature(aka. X) values
        for f in features:
            feature_list.append(list(f[1:]))
        features = np.asarray(features)
        # Get the matching score for every text
        for text_id in features[:, 0]:
            cur.execute(get_questionid_sql, text_id)
            question_id = cur.fetchone()[0]
            get_score_sql = "SELECT z" + str(question_id) + " FROM scores, detection WHERE detection.textid=%s " \
                            "and scores.studentid=detection.studentid"
            cur.execute(get_score_sql, text_id)
            score = cur.fetchone()[0]
            # print("current textid:", text_id, "question id:", question_id, "score:", score)
            score_list.append(score)
            score_text[text_id] = score
        # print(score_text)
        return feature_list, score_list
    except Exception:
        print("Error getting features...", traceback.print_exc())
    finally:
        cur.close()
        conn.close()


def cor_of_features(features, scores):
    """
    Paras:
        features:
            the matrix of feature values, shape of which is (M, N).
                --M is the number of samples and N is the number of features(7 for now).
                --features[i][j] is the NO.j feature value of NO.i sample.
            feature sequence is ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade', 'vecsim'].
        scores:
            matrix of scores, shape of which is (M, 1).
                --M is the number of samples.
    returns:
        A dict, key of which is feature name and value is pearson correlation value.
    """
    cors = {}
    i = 0
    features_arr = np.asarray(features)
    for f in ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade', 'vec']:
        cors[f] = round(pearson_cor(scores, features_arr[:, i]), 4)
        i += 1
    print(cors)
    return cors


# if __name__ == '__main__':
#     get_features()
#     feature, score = extract_features()
#     # print(feature)
#     # print(score)
#     # print(len(score))
#     cor_of_features(feature, score)

