import pickle
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ChatBot:
    def __init__(self, model_path):
        self.text = None
        self.tokenizer = Tokenizer()
        self.sequences = None
        self.max_sequence_len = None
        self.num_words = None
        self.input_sequences = None
        self.output_sequences = None
        self.model = None
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.tokenizer_path = self.model_path + "/tokenizer.data"

    def load_novel(self, novel_path):
        with open(novel_path, 'r') as f:
            self.text = f.read()

    def preprocess_text(self):
        self.tokenizer.fit_on_texts([self.text])
        self.sequences = self.tokenizer.texts_to_sequences([self.text])[0]
        self.input_sequences = []
        self.output_sequences = []
        for i in range(1, len(self.sequences)):
            self.input_sequences.append(self.sequences[:i])
            self.output_sequences.append(self.sequences[i])

        self.input_sequences = pad_sequences(self.input_sequences, maxlen=self.max_sequence_len, padding='pre')
        self.output_sequences = tf.keras.utils.to_categorical(self.output_sequences, num_classes=self.num_words)
        self.max_sequence_len = max([len(seq) for seq in self.input_sequences])
        self.num_words = len(self.tokenizer.word_index) + 1

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.num_words, 64, input_length=self.max_sequence_len),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
            tf.keras.layers.Dense(self.num_words, activation='softmax')
        ])

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, epochs=100):
        self.model.fit(self.input_sequences, self.output_sequences, epochs=epochs)

    def save_tokenizer(self):
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def load_tokenizer(self):
        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def save_model(self):
        self.model.save(self.model_path)
        self.save_tokenizer()

    def load_model(self):
        self.load_tokenizer()
        self.model = tf.keras.models.load_model(self.model_path)

    def generate_text(self, seed_text, next_words=20):
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.model.input.shape[1], padding='pre')
            predicted = self.model.predict(token_list, verbose=0)
            predicted_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == tf.argmax(predicted, axis=1).numpy()[0]:
                    predicted_word = word
                    break
            seed_text += " " + predicted_word
        return seed_text
