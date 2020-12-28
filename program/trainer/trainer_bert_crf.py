import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from program.data_process.data_preprocessor import GeneralDataPreprocessor
from program.models.model_bert_crf import BertCrfModel
from program.utils.custom_metrics import Metrics
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
from program.abstracts.abstract_ner_trainer import NerTrainer
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class BertCrfTrainer(NerTrainer):
    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            "ckiplab/bert-base-chinese-ner", from_pt=True, output_hidden_states=True
        )

    def transformer_tokenize(self, text):
        return self.tokenizer(
            text,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )

    def encode_tags(self, tags):
        unaligned_labels = [[self.tag2id[tag] for tag in content] for content in tags]
        token_o = self.tag2id["O"]
        bert_token_unaligned_labels = map(
            lambda labels: [token_o, *labels, token_o], unaligned_labels
        )

        return tf.keras.preprocessing.sequence.pad_sequences(
            list(bert_token_unaligned_labels),
            value=-1,
            padding="post",
            truncating="post",
            maxlen=self.max_sentence_length,
        )

    def tokenize(self):
        train_X, train_y = GeneralDataPreprocessor.loadTrainArrays(
            self.model_data_path + "train_X.pkl", self.model_data_path + "train_y.pkl"
        )
        self.unique_tags = set(tag for tags in train_y for tag in tags)
        self.tag2id = {tag: id for id, tag in enumerate(self.unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

        with open(self.model_data_path + "id2tag.pkl", "wb") as f:
            pickle.dump(self.id2tag, f)

        train_X, val_X, train_y, val_y = train_test_split(
            train_X, train_y, test_size=0.1
        )

        train_encodings = self.transformer_tokenize(train_X)
        val_encodings = self.transformer_tokenize(val_X)

        train_labels = self.encode_tags(train_y)
        val_labels = self.encode_tags(val_y)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(train_encodings), train_labels)
        )

        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(val_encodings), val_labels)
        )

    def train(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model = BertCrfModel(self.model, self.max_sentence_length, len(self.id2tag))
        model.compile(optimizer=optimizer)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            moniter="val_f1",
            mode="max",
        )

        self.history = model.fit(
            self.train_dataset.shuffle(1000).batch(self.batch_size),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=self.val_dataset.batch(self.batch_size),
            callbacks=[Metrics(self.val_dataset), checkpoint],
        )

    def visualize(self):
        plt.plot(self.history.history["val_f1"])
        plt.title("Model f1")
        plt.ylabel("f1")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()
