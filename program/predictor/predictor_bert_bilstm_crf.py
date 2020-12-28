import pickle
import tensorflow as tf
import numpy as np
import tensorflow_addons as tf_ad
from program.utils.write_output_file import format_result
from program.data_process.data_preprocessor import GeneralDataPreprocessor
from program.abstracts.abstract_ner_predictor import NerPredictor
from transformers import BertTokenizer, TFBertModel
from dataclasses import dataclass


@dataclass
class BertBilstmCrfPredictor(NerPredictor):
    def __post_init__(self):
        bert_model_name = [
            "hfl/chinese-bert-wwm",
            "hfl/chinese-bert-wwm-ext",
            "hfl/chinese-roberta-wwm-ext",
            "chinese-roberta-wwm-ext-large",
        ]

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name[0])
        self.bert_model = TFBertModel.from_pretrained(bert_model_name[0], from_pt=True)

        test_X_path = self.model_data_path + "test_X.pkl"
        test_mapping_path = self.model_data_path + "test_mapping.pkl"
        id2tag_path = self.model_data_path + "id2tag.pkl"

        test_X, self.test_mapping = GeneralDataPreprocessor.loadTestArrays(
            test_X_path, test_mapping_path
        )

        with open(id2tag_path, "rb") as f:
            self.id2tag = pickle.load(f)

        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_path))

    def predict_sentence(self, sentence):
        # dataset = encode sentence
        #         = [[1445   33 1878  826 1949 1510  112]]
        dataset = self.tokenizer(
            sentence,
            add_special_tokens=False,
            return_token_type_ids=False,
            is_split_into_words=True,
            padding=True,
        )
        
        dataset = tf.data.Dataset.from_tensors(dict(dataset)).batch(1)

        # logits = (1, 7, 28) = (sentence, words, predict_distrib)
        # text_lens = [7]
        logits, text_lens = self.model.predict(dataset)
        paths = []
        logits = logits.squeeze()[np.newaxis, :]
        text_lens = [sum(text_lens)]

        for logit, text_len in zip(logits, text_lens):
            viterbi_path, _ = tf_ad.text.viterbi_decode(
                logit[:text_len], self.model.transition_params
            )

            paths.append(viterbi_path)

        # paths[0] = tag in sentence
        #          = [18, 19, 19, 1, 26, 27, 1]

        # result  = ['B-name', 'I-name', 'I-name', 'O', 'B-time', 'I-time', 'O']
        result = [self.id2tag[id] for id in paths[0]]
        # entities_result =
        # [{'begin': 0, 'end': 3, 'words': '賈伯斯', 'type': 'name'},
        #  {'begin': 4, 'end': 6, 'words': '七號', 'type': 'time'}]
        entities_result = format_result(list(sentence), result)
        return entities_result

    def predict(self):
        # restore model
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_path))

        article_id = 0
        counter = 0
        results = []
        result = []
        for testset in self.test_X:
            prediction = self.predict_sentence(testset)

            # predict_pos + counter
            if prediction:
                for pred in prediction:
                    pred["begin"] += counter
                    pred["end"] += counter
                    result.append(pred)

            counter += len(testset)

            if counter == self.test_mapping[article_id]:
                results.append(result)
                article_id += 1
                counter = 0
                result = []

        self.results = results

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
                    elif (
                        self.id2tag[pred][0] == "O" or i + 1 == self.max_sentence_length
                    ):
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
