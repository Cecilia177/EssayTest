import jieba.analyse as analyse

ss = "我们不必一定去学习如何做到心理健康，这种能力植根于我们自身，就像我们的身体知道如何愈合伤口，如何修复断骨。"
print(" ".join(analyse.extract_tags(ss, topK=5, withWeight=False, allowPOS=())))






