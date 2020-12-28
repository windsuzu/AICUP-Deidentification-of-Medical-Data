import os
import tensorflow as tf


def read_vocab(vocab_file):
    """
    讀取 vocab_file，回傳 vocab <-> id 雙向的字典

    Args:
        vocab_file:
            input_vocab_file_path

    Returns:
        vocab2id:
            {'<UKN>': 0,
             '扎': 1,
             '遺': 2,
             ...}

        id2vocab:
            {0: '<UKN>',
             1: '扎',
             2: '遺',
             ...}
    """
    vocab2id = {}
    id2vocab = {}
    for index, line in enumerate(
        [line.strip() for line in open(vocab_file, "r", encoding="UTF8").readlines()]
    ):
        vocab2id[line] = index
        id2vocab[index] = line
    return vocab2id, id2vocab


def build_vocab(corpus_file_list, vocab_file, tag_file):
    """
    利用 train, test 建立 vocab file 和 tag file

    Args:
        corpus_file_list:
            [text_file_path]

        vocab_file:
            output_vocab_file_path

        tag_file:
            output_tag_file_path

    Output:
        vocab_file.txt
        tag.txt
    """
    words = set()
    tags = set()
    for file in corpus_file_list:
        for line in open(file, "r", encoding="utf-8").readlines():
            if line == "\n":
                continue
            try:
                w, t = line.split(" ")
                words.add(w)
                tags.add(t.strip("\n"))
            except Exception as e:
                raise e

    if not os.path.exists(vocab_file):
        with open(vocab_file, "w", encoding="utf-8") as f:
            for index, word in enumerate(["<UKN>"] + list(words)):
                f.write(word + "\n")

    tag_sort = {
        "O": 0,
        "B": 1,
        "I": 2,
        "E": 3,
    }

    tags = sorted(
        list(tags),
        key=lambda x: (
            len(x.split("-")),
            x.split("-")[-1],
            tag_sort.get(x.split("-")[0], 100),
        ),
    )
    if not os.path.exists(tag_file):
        with open(tag_file, "w", encoding="utf-8") as f:
            for index, tag in enumerate(["<UKN>"] + tags):
                f.write(tag + "\n")


def tokenize(train_X, train_y, vocab2id, tag2id, maxlen):
    """
    給定 train_X, train_y 輸出 tokenize 過的列表

    Args:
        vocab2id:
            voc -> id dictionary

        tag2id:
            tag -> id dictionary

    Returns:
        contents:
            array([[ 158, 2076, 1865, ...,    0,    0,    0],
                   [ 158, 2076, 1865, ...,    0,    0,    0],
                   [1019, 1023, 1865, ...,    0,    0,    0],
                   ...,
                   [ 158, 2076, 1865, ...,    0,    0,    0]])

        labels:
            array([[1, 1, 1, ..., 0, 0, 0],
                   [1, 1, 1, ..., 0, 0, 0],
                   [1, 1, 1, ..., 0, 0, 0],
                   ...,
                   [1, 1, 1, ..., 0, 0, 0]])
    """
    train_X = [[vocab2id[text] for text in X] for X in train_X]
    train_y = [[tag2id[label] for label in y] for y in train_y]
    
    train_X = tf.keras.preprocessing.sequence.pad_sequences(
        train_X, padding="post", maxlen=maxlen
    )
    train_y = tf.keras.preprocessing.sequence.pad_sequences(
        train_y, padding="post", maxlen=maxlen
    )
    return train_X, train_y

