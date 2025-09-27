import pandas as pd
import numpy as np
import jieba
from collections import Counter

# 2.读取数据
content = pd.read_csv('./data/train.txt', encoding='utf-8', sep='\t', names=['sentence', 'label'])
# print(content.head())

# 3.统计类别数量
counter = Counter(content['label'])
# print(counter)

# 样本数据分析
total = 0
for k, v in counter.items():
    total += v

# 样本类别的比例
for k, v in counter.items():
    print(k, (v / total)*100, '%')

# 文本长度分析
content['sentence_len'] = content['sentence'].apply(lambda x: len(x))
content_mean = np.mean(content['sentence_len'])
content_std = np.std(content['sentence_len'])
print(content.head())
