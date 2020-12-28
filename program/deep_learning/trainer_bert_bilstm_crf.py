import tensorflow as tf
import tensorflow_addons as tf_ad
import matplotlib.pyplot as plt
from program.deep_learning.model_bert_bilstm_crf import BertBilstmCrfModel
from program.deep_learning.data_preprocessor import GeneralDataPreprocessor
from transformers import BertTokenizer, TFBertModel
from program.abstracts.abstract_ner_trainer import NerTrainer
from dataclasses import dataclass


@dataclass
class BertBilstmCrfTrainer(NerTrainer):
    def __post_init__(self):
        bert_model_name = [
            "hfl/chinese-bert-wwm",
            "hfl/chinese-bert-wwm-ext",
            "hfl/chinese-roberta-wwm-ext",
            "chinese-roberta-wwm-ext-large",
        ]

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name[0])
        self.bert_model = TFBertModel.from_pretrained(bert_model_name[0], from_pt=True)

    def encode_tags(self, tags):
        return tf.keras.preprocessing.sequence.pad_sequences(
            [[self.tag2id[label] for label in tag] for tag in tags],
            value=self.tag2id["O"],
            padding="post",
            truncating="post",
            maxlen=self.max_sentence_length,
        )

    def tokenize(self):
        train_X, train_y = GeneralDataPreprocessor.loadTrainArrays(
            self.model_data_path + "train_X.pkl", self.model_data_path + "train_y.pkl"
        )

        unique_tags = set(tag for tags in train_y for tag in tags)
        self.tag2id = {tag: id for id, tag in enumerate(unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

        train_encodings = self.tokenizer(
            train_X,
            add_special_tokens=False,
            return_token_type_ids=False,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_sentence_length,
        )

        train_labels = self.encode_tags(train_y)
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
            .shuffle(len(train_X))
            .batch(self.batch_size, drop_remainder=True)
        )

    def train(self):
        model = BertBilstmCrfModel(
            bert_model=self.bert_model,
            hidden_num=self.hidden_nums,
            label_size=len(self.tag2id),
        )

        # freeze BERT embedding parameters
        model.layers[0].trainable = False

        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            self.checkpoint_path,
            checkpoint_name="model.ckpt",
            max_to_keep=self.checkpoint_keep,
        )
        
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
                accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
            accuracy = accuracy / len(paths)
            return accuracy
        
        best_acc = 0
        step = 0
        self.loss_history = []

        for epoch in range(self.epochs):
            for _, (text_batch, labels_batch) in enumerate(self.train_dataset):
                step += 1
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