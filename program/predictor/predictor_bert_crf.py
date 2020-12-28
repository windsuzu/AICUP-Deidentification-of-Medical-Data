import pickle
from program.data_process.data_preprocessor import GeneralDataPreprocessor
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
from program.abstracts.abstract_ner_predictor import NerPredictor


@dataclass
class BertCrfPredictor(NerPredictor):
    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            "ckiplab/bert-base-chinese-ner", from_pt=True, output_hidden_states=True
        )

        test_X_path = self.model_data_path + "test_X.pkl"
        test_mapping_path = self.model_data_path + "test_mapping.pkl"
        id2tag_path = self.model_data_path + "id2tag.pkl"

        test_X, self.test_mapping = GeneralDataPreprocessor.loadTestArrays(
            test_X_path, test_mapping_path
        )

        with open(id2tag_path, "rb") as f:
            self.id2tag = pickle.load(f)

        test_encodings = self.tokenizer(
            test_X,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )

        self.test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings)))

    def predict(self):
        prediction = self.model.predict(self.test_dataset)[0]
        prediction = np.argmax(prediction, -1)
        self.prediction = prediction.reshape(-1, self.max_sentence_length)

    def output(self):
        output = []

        article_id = 0
        start_batch = 0
        end_batch = 0

        for article in self.test_mapping:
            start_batch = end_batch
            end_batch += (len(article) // self.max_sentence_length) + 1

            pos_counter = 0
            entity_type = None
            start_pos = None
            end_pos = None

            for preds in self.prediction[start_batch:end_batch]:
                # get rid of [CLS], [SEP] in common batches
                # exceptions only occur in last batches, no matters
                preds = preds[1:-1]

                for i, pred in enumerate(preds):
                    if self.id2tag[pred][0] == "B":
                        start_pos = pos_counter
                        entity_type = self.id2tag[pred][2:]  # remove "B-"
                    elif self.id2tag[pred][0] == "I":
                        end_pos = pos_counter
                    elif self.id2tag[pred][0] == "O" or i + 1 == self.max_sentence_length:
                        if entity_type:
                            entity_name = article[start_pos : (end_pos + 1)]
                            output.append(
                                (
                                    article_id,
                                    start_pos,
                                    end_pos,
                                    entity_name,
                                    entity_type,
                                )
                            )
                            entity_type = None
                    pos_counter += 1
