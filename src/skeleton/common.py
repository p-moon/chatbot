

import nltk
from collections import Counter
import numpy as np


def preprocess_corpus(corpus):
    """
    对语料库进行预处理，包括分词、去除停用词等
    :param corpus:
    :return:
    """
    # 分词
    tokens = [nltk.word_tokenize(sentence) for sentence in corpus]
    # 去除停用词
    stop_words = nltk.corpus.stopwords.words("chinese")
    filtered_tokens = [[word for word in sentence if word not in stop_words] for sentence in tokens]
    return filtered_tokens


def build_dictionary(processed_corpus):
    """
    创建词典
    :param processed_corpus:
    :return:
    """
    # 统计词频
    word_freq = Counter([word for sentence in processed_corpus for word in sentence])
    # 按词频从大到小排序
    sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    # 创建词典
    word_to_idx = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    num_word_tokens = len(word_to_idx)
    return word_to_idx, idx_to_word, num_word_tokens


def build_training_data(corpus_indices, num_word_tokens, batch_size):
    """
    将语料库转换为数字序列
    :param corpus_indices:
    :param num_word_tokens:
    :param batch_size:
    :return:
    """
    num_batches = (len(corpus_indices) - 1) // batch_size
    train_data = []
    for i in range(num_batches):
        data = corpus_indices[i * batch_size:(i + 1) * batch_size]
        target = np.zeros((batch_size, num_word_tokens))
        for j, index in enumerate(data):
            target[j][index] = 1
        train_data.append((data, target))
    return train_data


