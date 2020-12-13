# %%
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path


def getTrainData(root_datapath):

    """
    Load crf's train.data to get the train data in the format of the transformers.

    Args:
        root_datapath:
            dataset/crf_data/train.data

    Returns:
        train_texts (dataset_size, content_size):
            [...[...'：', '阿', '只', '是', '前', '天', '好', '很', '多'...]...]

        train_tags (dataset_size, content_size):
            [...[...'O', 'O', 'O', 'O', 'B-time', 'I-time', 'O', 'O', 'O'...]...]
    """
    train_texts = []
    train_tags = []
    path = Path(__file__).parents[2] / root_datapath
    with path.open(encoding="UTF8") as f:
        train_data = f.readlines()
        texts = []
        tags = []

        # line format: 中 B-name\n
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


def getTestData(root_datapath):
    """Load crf's test.data to get the test data in the format of the transformers.

    Args:
        root_datapath:
            dataset/crf_data/test.data

    Returns:
        test_texts (dataset_size, content_size):
            [...[...'有', '辦', '法', '，', '這', '是', '嚴', '重', '或', '一', ...]...]
    """
    test_texts = []
    path = Path(__file__).parents[2] / root_datapath
    with path.open(encoding="UTF8") as f:
        test_data = f.readlines()
        texts = []

        for line in test_data:
            if line == "\n":
                test_texts.append(texts)
                texts = []
                continue

            text = line[0]
            texts.append(text)

    return test_texts


def encode_tags(tags, encodings):
    unaligned_labels = [[tag2id[tag] for tag in content] for content in tags]
    max_length = len(encodings.input_ids[0])

    return tf.keras.preprocessing.sequence.pad_sequences(
        unaligned_labels, value=-1, padding="post", truncating="post", maxlen=max_length
    )


#%%
train_texts, train_tags = getTrainData("dataset/crf_data/train.data")
test_texts = getTestData("dataset/crf_data/test.data")

unique_tags = set(tag for tags in train_tags for tag in tags)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


#%%
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_tags, val_tags = train_test_split(
    train_texts, train_tags, test_size=0.1
)


# %%
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")
model = AutoModelForTokenClassification.from_pretrained("ckiplab/bert-base-chinese-ner")

# %%
train_encodings = tokenizer(
    train_texts, is_split_into_words=True, padding=True, truncation=True
)
val_encodings = tokenizer(
    val_texts, is_split_into_words=True, padding=True, truncation=True
)

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

#%%
train_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(train_encodings), train_labels)
)

val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))


# %%
# TODO: checkpoint training

# TODO: graphing training

# TODO: predicting

# TODO: uploading
