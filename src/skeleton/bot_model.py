import tensorflow as tf
import numpy as np
import pickle
import os
import json

class ChatBot:
    def __init__(self, model_dir, filename=None, seq_length=100, batch_size=64, encoding='utf-8'):
        self.filename = filename
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.encoding = encoding

        self.chars = None
        self.vocab_size = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.data = None
        self.steps_per_epoch = None

        self.text = None

        self.model_dir = model_dir
        self.check_point_dir = self.model_dir + "/check_point"
        self.model = tf.keras.Sequential()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.check_point_dir):
            os.makedirs(self.check_point_dir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def load_text(self):
        with open(self.filename, 'r', encoding=self.encoding) as f:
            self.text = f.read()
        self._encode_text()

    def _encode_text(self):
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.data = np.array([self.char_to_idx[c] for c in self.text])
        self.steps_per_epoch = (len(self.text) - self.seq_length) // self.batch_size

    def _create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.data)
        dataset = dataset.window(self.seq_length + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.seq_length + 1))
        dataset = dataset.map(lambda window: (window[:-1], window[1:]))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def _build_model(self, embedding_dim=256, rnn_units=1024):
        dataset = self._create_dataset()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, embedding_dim, batch_input_shape=[self.batch_size, None]),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(self.vocab_size)
        ])
        self.model.summary()
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        return self.model

    def save(self):
        char_to_idx_path = self.model_dir + "/char_to_idx.data"
        idx_to_char_path = self.model_dir + "/idx_to_char.data"
        self.model.save(self.model_dir)
        with open(char_to_idx_path, "wb") as f:
            pickle.dump(self.char_to_idx, f)
        with open(idx_to_char_path, "wb") as f:
            pickle.dump(self.idx_to_char, f)

    @classmethod
    def load(cls, model_path, seq_length=100, batch_size=64):
        char_to_idx_path = model_path + "/char_to_idx.data"
        idx_to_char_path = model_path + "/idx_to_char.data"
        with open(char_to_idx_path, "rb") as f:
            char_to_idx = pickle.load(f)
        with open(idx_to_char_path, "rb") as f:
            idx_to_char = pickle.load(f)
        model = tf.keras.models.load_model(model_path)
        generator = cls(model_dir=model_path, seq_length=seq_length, batch_size=batch_size)
        generator.char_to_idx = char_to_idx
        generator.idx_to_char = idx_to_char
        generator.model = model
        return generator

    def train(self, epochs=10, embedding_dim=256, rnn_units=1024, num_threads=4):
        dataset = self._create_dataset()
        self._build_model(embedding_dim=embedding_dim, rnn_units=rnn_units)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.repeat(epochs)
        num_batches = (len(self.text) - 1) // self.batch_size
        steps_per_epoch = num_batches // self.batch_size
        self.model.fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[CheckpointCallback(self.check_point_dir)],
            verbose=1,
            max_queue_size=num_threads * 10,
            workers=num_threads,
            use_multiprocessing=True
        )

    def generate_text(self, start_string, temperature=1.0, num_generate=1000):
        input_eval = [self.char_to_idx[c] for c in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        self.model.reset_states()
        for i in range(num_generate):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx_to_char[predicted_id])

        return start_string + ''.join(text_generated)


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, save_freq=1):
        super(CheckpointCallback, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            self.model.save_weights(os.path.join(self.checkpoint_dir, f"ckpt_{epoch + 1}"))
            print(f"Model saved at epoch {epoch + 1}" + json.dumps(logs))
