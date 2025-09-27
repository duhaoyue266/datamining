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

# 样本总数
total = sum(counter.values())

# 样本类别的比例
# for k, v in counter.items():
#     print(k, (v / total)*100, '%')

# 文本长度分析
content['sentence_len'] = content['sentence'].apply(lambda x: len(x))
content_mean = np.mean(content['sentence_len'])
content_std = np.std(content['sentence_len'])
# print(content.head())

# jieba分词
def cut_sentence(s):
    return jieba.lcut(s)

# 分词结果（list）
content['words'] = content['sentence'].apply(cut_sentence)

# 保留前30个词，并拼接成字符串
content['words'] = content['words'].apply(lambda x: ' '.join(x[:30]))

# 保存结果
content.to_csv('./data/train_new.txt', index=False, encoding='utf-8')
