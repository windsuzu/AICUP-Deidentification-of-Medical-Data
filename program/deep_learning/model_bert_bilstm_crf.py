import tensorflow as tf
import tensorflow_addons as tf_ad


class BertBilstmCrfModel(tf.keras.Model):
    def __init__(self, bert_model, hidden_num, label_size):
        super(BertBilstmCrfModel, self).__init__()

        self.embedding = bert_model

        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_num, return_sequences=True)
        )
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(label_size)

        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(label_size, label_size))
        )

    @tf.function
    def call(self, text, labels=None, training=None):
        text_lens = tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(text["input_ids"], 0), dtype=tf.int32), axis=-1
        )
        # -1 change 0
        # 64, 32, 768
        embedding_layer = self.embedding(
            text["input_ids"], attention_mask=text["attention_mask"]
        )[0]

        inputs = self.dropout(embedding_layer, training)
        # 64, 32, 27
        logits = self.dense(self.biLSTM(inputs))

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(
                logits,
                label_sequences,
                text_lens,
                transition_params=self.transition_params,
            )
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens