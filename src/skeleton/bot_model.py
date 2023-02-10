import tensorflow as tf
import numpy as np
from skeleton.common import *

"""
batch_size：是批量训练的大小，它指定了在每次训练迭代中网络要看到的数据样本数量。如果将 batch_size 设置为 128，则网络每次迭代都将看到 128 个数据样本。
embedding_dimension：是词嵌入的维数。词嵌入是一种将词汇表示为密集矢量（即浮点数数组）的方法，这些矢量表示词语的语义。这个参数控制着词嵌入的维数，比如如果设为 64，则词嵌入的每个词的向量大小为 64 位。
num_lstm_units：是 LSTM 单元的数量，LSTM 是一种递归神经网络（RNN）
"""
batch_size = 128
embedding_dimension = 64
num_lstm_units = 64
num_word_tokens = 10000


class ChatBot:
    """模型对象"""
    model = None

    """模型存储&加载地址"""
    model_dir = None

    """语料库"""
    corpus = []

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.init_model()

    def init_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=num_word_tokens, output_dim=embedding_dimension,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(units=num_lstm_units, return_sequences=True, stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(units=num_word_tokens)
        ])

    def compile_model(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    def add_material(self, content):
        self.corpus.append(content)

    def add_material_from_file(self, file_path: str):
        input_corpus = open(file_path, "r").read().splitlines()
        for content in input_corpus:
            self.corpus.append(content)

    def train(self):
        processed_corpus = preprocess_corpus(self.corpus)

        # 创建词典，并对语料库进行编号
        word_to_idx, idx_to_word, num_word_tokens = build_dictionary(processed_corpus)

        # 将语料库转换为数字序列
        corpus_indices = [word_to_idx[word] for sentence in processed_corpus for word in sentence]

        # 准备训练数据
        train_data = build_training_data(corpus_indices, num_word_tokens, batch_size)

        # 训练模型
        self.model.fit(train_data, epochs=5)

    def load_model(self):
        self.model = tf.keras.models.load_model()

    def save_model(self):
        self.model.save(self.model_dir)

    def predict(self, input_sentence: str) -> str:
        return self.model.predict(input_sentence)
