import pymysql
from sentence import Sentence
from LSA import LSA
import traceback
import numpy as np
from correlation import pearson_cor
from vecsim import vector_similarity
from gensim.models import KeyedVectors
from fluency import get_fluency_score
from stanfordcorenlp import StanfordCoreNLP
from partofspeech import get_phrases

pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
pymysql.converters.conversions = pymysql.converters.encoders.copy()
pymysql.converters.conversions.update(pymysql.converters.decoders)


def cal_features(conn):
    """
    Calculate features of texts of all courses and questions and insert those into DB.
    To put it simple, complete the features table in DB.
    Parameters:
        conn: A mysql connection.

    """

    # Get coureseid, questionid and reference of all courses.
    try:
        get_cour_sql = "SELECT courseid, questionid FROM standards"
        get_cour_cur = conn.cursor()
        get_cour_cur.execute(get_cour_sql)
        course = get_cour_cur.fetchall()
    except Exception as e:
        print("Error getting courses and questions!", e)
    finally:
        get_cour_cur.close()

    word_vectors = KeyedVectors.load("./model/vectors.kv")
    record_count = 0
    zh_model = StanfordCoreNLP(r"H:\Download\stanford-corenlp-full-2018-02-27", lang='zh')

    for courseid, questionid in course:
        print("--------------------current question:", questionid, "of", courseid, "----------------------")
        if get_docs_list(conn=conn, courseid=courseid, questionid=questionid) is None:
            continue

        textids, doc_matrix = get_docs_list(conn=conn, courseid=courseid, questionid=questionid)
        mylsa = build_svd(doc_matrix)
        reference = doc_matrix[0]  # reference is Class Sentence and is preprocessed already.
        ref_np_length = get_phrases(phrase="NP", sentence=reference, model=zh_model)
        ref_vp_length = get_phrases(phrase="VP", sentence=reference, model=zh_model)
        for i in range(1, len(doc_matrix)):
            current_answer = doc_matrix[i]
            textid = textids[i]
            print("This is NO.", i, "text with textid--", textid)

            # Calculate features including LENGTHRATIO, 1~4GRAM, LSAGRADE, VEC_SIM
            lengthratio = format(float(current_answer.seg_length) / reference.seg_length, '.2f')
            bleu = get_bleu_score(reference, current_answer)
            lsagrade = mylsa.get_similarity(10, 0, i)
            vec_sim = vector_similarity(id1=0, id2=i, vecs=word_vectors, stopwords=[], tf_idf=mylsa.A, keys=mylsa.keys)
            fluency = get_fluency_score(ref=reference, sentence=current_answer)
            np_length = get_phrases(phrase="NP", sentence=current_answer, model=zh_model)
            vp_length = get_phrases(phrase="VP", sentence=current_answer, model=zh_model)
            # Inserting features of a certain text into DB
            if insert_features(conn=conn, textid=textid, ngram=bleu, lengthratio=lengthratio,
                               lsagrade=lsagrade, vec_sim=vec_sim, fluency=fluency,
                               np=np_length/ref_np_length, vp=vp_length/ref_vp_length):
                record_count += 1
    print("--------------------Finishing inserting features of", record_count, "text.----------------------")


def insert_features(conn, textid, ngram, lengthratio, lsagrade, vec_sim, fluency, np, vp):
    """
    Insert a feature record into DB.
    Parameters:
        conn: A mysql connection.
        textid, ngram, lengthratio, lsagrade, vec_sim: Features to insert into DB
    Returns:
        Boolean, true if success inserting else false.
    """
    try:
        insert_feature_sql = "INSERT INTO features(textid, 1gram, 2gram, 3gram, 4gram, lengthratio, lsagrade, vecsim, fluency, np, vp)" \
                             "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        insert_feature_cur = conn.cursor()
        print("Inserting features__textid:", textid, "1~4gram:", ngram, "lengthratio:", lengthratio,
              "lsa:", lsagrade, "vec:", vec_sim, "fluency:", fluency, "np:", np, "vp:", vp)
        insert_feature_cur.execute(insert_feature_sql, (
            textid, ngram[0], ngram[1], ngram[2], ngram[3], lengthratio, lsagrade, vec_sim, fluency, np, vp))
        conn.commit()
        print("Success!")
        return True
    except Exception as e:
        print("Error inserting features..", traceback.print_exc())
        conn.rollback()
        return False
    finally:
        insert_feature_cur.close()


def build_svd(docs_list):
    """
    Parameters:
        A list of docs.
    Returns:
        A LSA object, containing matrix A as tf-idf matrix,
            and method get_similarity() to cal the similarity between 2 docs in docs_list.
    """
    # build count matrix, tf-idf modification matrix and get svd.
    lsa = LSA(stopwords=[], ignorechars="")
    for doc in docs_list:
        lsa.parse(doc)
    lsa.build_count_matrix()
    lsa.TFIDF()
    lsa.svd_cal()
    return lsa


def get_docs_list(conn, courseid, questionid):
    """
    Parameters:
        conn: A mysql connection.
        courseid: String
        questionid: Integer
    Return:
        Two lists: list of textid and list of doc(class Sentence), the first terms of both are -1 and Sentence reference.
    """
    # Get reference of certain courseid and questionid.
    try:
        get_ref_sql = "SELECT ref FROM standards WHERE courseid=%s AND questionid=%s"
        get_ref_cur = conn.cursor()
        get_ref_cur.execute(get_ref_sql, (courseid, questionid))
        ref = get_ref_cur.fetchone()[0]

    except Exception as e:
        print("Error getting reference of courseid", courseid, "questionid", questionid)
        print(traceback.print_exc())
    finally:
        get_ref_cur.close()

    reference = Sentence(text=ref, language="ch")
    reference.preprocess()
    doc_matrix = [reference]  # add Sentence reference as the first term of doc_matrix
    textids = [-1]  # Use -1 as the referece textid

    # Get all detection text of certain courseid and questionid.
    try:
        get_detection_sql = "SELECT textid, text FROM detection WHERE courseid = %s and questionid = %s"
        get_detection_cur = conn.cursor()
        if get_detection_cur.execute(get_detection_sql, (courseid, questionid)):
            detections = get_detection_cur.fetchall()
        else:
            detections = None
            print("No quesion", questionid, "of", courseid, "in DETECTION DB.")
    except Exception as e:
        print("Error getting text...", traceback.print_exc())
    finally:
        get_detection_cur.close()

    # Add all detections into doc_matrix
    if detections == None:
        return
    for dt in detections:
        textids.append(dt[0])
        cur_ans = Sentence(text=dt[1], language="ch")
        cur_ans.preprocess()
        doc_matrix.append(cur_ans)
    return textids, doc_matrix


def get_bleu_score(ref, answer):
    """
    paras:
        ref: class Sentence
        answer: class Sentence
    Return:
        A list including 1~4gram matching rate of answer compared to ref,
            eg. [1gram rate, 2gram rate, 3gram rate, 4gram rate]
    """
    ref_ngram = ref.ngram
    answer_ngram = answer.ngram
    total_count = [0] * 4
    match_count = [0] * 4
    for key in answer_ngram.keys():
        n = len(key.split(" ")) - 1   # key is (n+1)gram
        total_count[n] += 1
        if key in ref_ngram.keys():
            match_count[n] += min(answer_ngram[key], ref_ngram[key])
            # print("Got one:", key, "+", min(answer_ngram[key], ref_ngram[key]))
    # bleu formula.
    # score = math.exp(sum([math.log(float(a)/b) for a, b in zip(match_count, total_count)]) * 0.25)
    score_list = []
    for i in range(4):
        score_list.append(format(float(match_count[i] / total_count[i]), ".4f"))
    # score_list.append(score)
    return score_list


def extract_data(conn):
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
    get_all_features_sql = "SELECT textid, 1gram, 2gram, 3gram, 4gram, lengthratio, lsagrade, vecsim, fluency, np, vp FROM features"
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
            # ss = "current textid:" + str(text_id) + " question id:" + str(question_id) + " score:" + str(score)
            # with open("C:\\Users\\Cecilia\\Desktop\\y.txt", "a") as f:
            #     f.write(ss)
            #     f.write("\n")
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
                --M is the number of samples and N is the number of features(8 for now).
                --features[i][j] is the NO.j feature value of NO.i sample.
            feature sequence is ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade', 'vecsim', 'fluency'].
        scores:
            matrix of scores, shape of which is (M, 1).
                --M is the number of samples.
    returns:
        A dict, key of which is feature name and value is pearson correlation value.
    """
    cors = {}
    i = 0
    features_arr = np.asarray(features)
    for f in ['1gram', '2gram', '3gram', '4gram', 'lengthratio', 'lsagrade', 'vec', 'fluency', 'np', 'vp']:
        cors[f] = round(pearson_cor(scores, features_arr[:, i]), 4)
        i += 1
    print(cors)
    return cors


if __name__ == '__main__':
    conn = pymysql.connect(host="127.0.0.1",
                           database='essaydata',
                           port=3306,
                           user='root',
                           password='',
                           charset='utf8')
    cal_features(conn)
    feature, score = extract_data(conn)
    # print(feature)
    # print(score)
    # print(len(score))
    cor_of_features(feature, score)
    # conn.close()

