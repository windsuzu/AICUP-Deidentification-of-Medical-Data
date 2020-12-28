from program.models.model_bilstm_crf import BilstmCrfModel
from program.data_process.data_preprocessor import GeneralDataPreprocessor
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tf_ad
from program.utils.tokenization import read_vocab
from dataclasses import dataclass
from program.utils.write_output_file import format_result
from program.abstracts.abstract_ner_predictor import NerPredictor


@dataclass
class BilstmCrfPredictor(NerPredictor):
    def __post_init__(self):
        vocab_file_path = self.model_data_path + "vocab_file.txt"
        tag_file_path = self.model_data_path + "tag.txt"

        self.voc2id, self.id2voc = read_vocab(vocab_file_path)
        self.tag2id, self.id2tag = read_vocab(tag_file_path)

        test_X_path = self.model_data_path + "test_X.pkl"
        test_mapping_path = self.model_data_path + "test_mapping.pkl"

        self.test_X, self.test_mapping = GeneralDataPreprocessor.loadTestArrays(
            test_X_path, test_mapping_path
        )

        self.model = BilstmCrfModel(
            hidden_num=self.hidden_nums,
            vocab_size=len(self.voc2id),
            label_size=len(self.tag2id),
            embedding_size=self.embedding_size,
        )
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def predict_sentence(self, sentence):
        """
        predict single sentence.

        Input:
            Raw text string
        """
        # dataset = encode sentence
        #         = [[1445   33 1878  826 1949 1510  112]]
        dataset = tf.keras.preprocessing.sequence.pad_sequences(
            [[self.voc2id.get(char, 0) for char in sentence]], padding="post"
        )

        # logits = (1, 7, 28) = (sentence, words, predict_distrib)
        # text_lens = [7]
        logits, text_lens = self.model.predict(dataset)
        paths = []

        for logit, text_len in zip(logits, text_lens):
            viterbi_path, _ = tf_ad.text.viterbi_decode(
                logit[:text_len], self.model.transition_params
            )
            paths.append(viterbi_path)

        # path[0] = tag in sentence
        #         = [18, 19, 19, 1, 26, 27, 1]

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
        """
        results:
            [
                [
                    {'begin': 170, 'end': 174, 'words': '1100', 'type': 'med_exam'},
                    {'begin': 245, 'end': 249, 'words': '1145', 'type': 'med_exam'},
                    ...
                ]
            ]
        """

        titles = {
            "end": "end_position",
            "begin": "start_position",
            "words": "entity_text",
            "type": "entity_type",
        }
        df = pd.DataFrame()

        for i, result in enumerate(self.results):
            results = pd.DataFrame(result).rename(columns=titles)
            results = results[
                ["start_position", "end_position", "entity_text", "entity_type"]
            ]

            article_ids = pd.Series([i] * len(result), name="article_id")
            df = df.append(pd.concat([article_ids, results], axis=1), ignore_index=True)

        df.to_csv(self.output_path + "output.tsv", sep="\t", index=False)
