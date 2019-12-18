from feature import extract_features
import numpy as np
import math
from correlation import pearson_cor


feature_list, score_list = extract_features()
new_scores = np.array(score_list) / 2.0
predict_scores = []
for f in feature_list:
    # print(f)
    length_ratio = f[4]   # candidate length / reference length
    if length_ratio == 0:
        BLEU = 0
    else:
        BP = 1 if length_ratio > 1 else math.exp(1 - 1 / length_ratio)
        BLEU = BP * math.exp(sum(math.log(p) for p in f[:4]) * 0.25) if f[0]*f[1]*f[2]*f[3] != 0 else 0
    predict_scores.append(BLEU)

print(new_scores)
print(predict_scores)
print(pearson_cor(new_scores, predict_scores))
