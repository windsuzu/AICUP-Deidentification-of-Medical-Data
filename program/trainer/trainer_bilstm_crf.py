from pathlib import Path
from program.models.model_bilstm_crf import BilstmCrfModel
from program.data_process.data_preprocessor import GeneralDataPreprocessor
from program.utils.tokenization import build_vocab, read_vocab
from program.utils.tokenization import tokenize as tk
import tensorflow as tf
import tensorflow_addons as tf_ad
import matplotlib.pyplot as plt
from program.abstracts.abstract_ner_trainer import NerTrainer
from dataclasses import dataclass


@dataclass
class BiLstmCrfTrainer(NerTrainer):
    def tokenize(self):
        vocab_file_path = self.model_data_path + "vocab_file.txt"
        tag_file_path = self.model_data_path + "tag.txt"

        if not Path(vocab_file_path).exists() and not Path(tag_file_path).exists():
            build_vocab([self.train_data_path], vocab_file_path, tag_file_path)

        self.voc2id, self.id2voc = read_vocab(vocab_file_path)
        self.tag2id, self.id2tag = read_vocab(tag_file_path)

        train_X, train_y = GeneralDataPreprocessor.loadTrainArrays(
            self.model_data_path + "train_X.pkl", self.model_data_path + "train_y.pkl"
        )

        train_X, train_y = tk(
            train_X, train_y, self.voc2id, self.tag2id, self.max_sentence_length
        )

        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_X, train_y))
            .shuffle(len(train_X))
            .batch(self.batch_size, drop_remainder=True)
        )

    def train(self):
        model = BilstmCrfModel(
            hidden_num=self.hidden_nums,
            vocab_size=len(self.voc2id),
            label_size=len(self.tag2id),
            embedding_size=self.embedding_size,
        )
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        @tf.function
        def train_one_step(text_batch, labels_batch):
            with tf.GradientTape() as tape:
                logits, text_lens, log_likelihood = model(
                    text_batch, labels_batch, training=True
                )
                loss = -tf.reduce_mean(log_likelihood)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, logits, text_lens

        def get_acc_one_step(logits, text_lens, labels_batch):
            paths = []
            accuracy = 0
            for logit, text_len, labels in zip(logits, text_lens, labels_batch):
                viterbi_path, _ = tf_ad.text.viterbi_decode(
                    logit[:text_len], model.transition_params
                )
                paths.append(viterbi_path)
                correct_prediction = tf.equal(
                    tf.convert_to_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(
                            [viterbi_path], padding="post"
                        ),
                        dtype=tf.int32,
                    ),
                    tf.convert_to_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(
                            [labels[:text_len]], padding="post"
                        ),
                        dtype=tf.int32,
                    ),
                )
                accuracy = accuracy + tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32)
                )

            accuracy = accuracy / len(paths)
            return accuracy

        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            self.checkpoint_path,
            checkpoint_name="model.ckpt",
            max_to_keep=self.checkpoint_keep,
        )

        best_acc = 0
        step = 0
        self.loss_history = []

        for epoch in range(self.epochs):
            for _, (text_batch, labels_batch) in enumerate(self.train_dataset):
                step = step + 1
                loss, logits, text_lens = train_one_step(text_batch, labels_batch)
                self.loss_history.append(loss)
                
                if step % 20 == 0:
                    accuracy = get_acc_one_step(logits, text_lens, labels_batch)
                    print(
                        f"epoch {epoch}, step {step}, loss {loss:.4f} , accuracy {accuracy:.4f}"
                    )

                    if accuracy > best_acc:
                        best_acc = accuracy
                        ckpt_manager.save()
                        print("model saved")

    def visualize(self):
        plt.plot(self.loss_history)
        plt.title("Model Loss")
        plt.ylabel("loss")
        plt.xlabel("Step")
        plt.legend(["Train"], loc="upper left")
        plt.show()
