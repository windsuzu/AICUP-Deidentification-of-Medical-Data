from pathlib import Path
import pickle
from program.utils.load_data import loadTestFile

import numpy as np
from program.abstracts.abstract_data_preprocessor import DataPreprocessor


def getTrainData(datapath):
    """
    Load crf's train.data to get the train data in the format below.

    Args:
        train.data with "\n" divides the articles:
            只 O\n
            是 O\n
            前 B-time\n
            天 I-time\n
            ...
            \n
            只 O\n

    Returns:
        train_texts (dataset_size, content_size):
            [
                ['只', '是', '前', '天', '好', '很', ...],
                ['只', '是', '前', '天', '好', '很', ...],
                ...
            ]

        train_tags (dataset_size, content_size):
            [
                ['O', 'O', 'B-time', 'I-time', 'O', 'O', ...],
                ['O', 'O', 'B-time', 'I-time', 'O', 'O', ...],
                ...
            ]
    """
    train_texts = []
    train_tags = []
    with Path(datapath).open(encoding="UTF8") as f:
        train_data = f.readlines()
        texts = []
        tags = []

        # content of each line: "中 B-name\n"
        # line[0] = 中
        # line[2:-1] = B-name
        for line in train_data:
            if line == "\n":
                train_texts.append(texts)
                train_tags.append(tags)
                texts = []
                tags = []
                continue

            text, tag = line[0], line[2:-1]
            texts.append(text)
            tags.append(tag)

    return train_texts, train_tags


def getTestData(datapath, raw_test_data_path):
    """Load crf's test.data to get the test data in the format of the transformers.

    Returns:
        test_texts (dataset_size, content_size):
            [
                ['有', '辦', '法', '，', '這', '是', '嚴', '重', '或', '一', ...],
                ['有', '辦', '法', '，', '這', '是', '嚴', '重', '或', '一', ...],
                ...
            ]

        test_mapping: test texts without split.
            [
                1234,
                342,
                1123,
                ...
            ]
    """
    test_mapping = [len(article) for article in loadTestFile(raw_test_data_path)]
    test_texts = []
    content = []
    with Path(datapath).open(encoding="UTF8") as f:

        for line in f.readlines():
            if line != "\n":
                w = line.strip("\n")
                content.append(w)
            else:
                if content:
                    test_texts.append("".join(content))
                content = []

    return test_texts, test_mapping


def remove_imbalance_trainsets(train_texts, train_tags, percent):
    o_sets = []
    trainsets = []
    for text, tag in zip(train_texts, train_tags):
        if all([token == "O" for token in tag]):
            if len(text) > 7:
                o_sets.append((text, tag))
        else:
            trainsets.append((text, tag))
    import random
    random.shuffle(o_sets)
    o_sets = o_sets[:int(len(o_sets) * percent)]
    trainsets.extend(o_sets)
    random.shuffle(trainsets)
    return zip(*trainsets)


class GeneralDataPreprocessor(DataPreprocessor):
    def __init__(self, train_data_path, test_data_path):
        super().__init__(train_data_path, test_data_path)

    def outputTrainArrays(self, train_X_path, train_y_path):
        train_X, train_y = getTrainData(self.train_data_path)

        with open(train_X_path, "wb") as f:
            pickle.dump(train_X, f)

        with open(train_y_path, "wb") as f:
            pickle.dump(train_y, f)

        print("Train text data & labels array saved")

    def outputTestArray(self, test_X_path, test_mapping_path, raw_test_data_path):
        test_X, mapping = getTestData(self.test_data_path, raw_test_data_path)

        with open(test_X_path, "wb") as f:
            pickle.dump(test_X, f)

        with open(test_mapping_path, "wb") as f:
            pickle.dump(mapping, f)

        print("Test text data and mapping array saved")