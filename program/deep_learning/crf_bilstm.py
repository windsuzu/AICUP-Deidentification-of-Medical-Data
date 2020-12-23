#%%
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path().resolve().parents[1]))

from program.utils.load_data import loadTestFile
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tf_ad
import os

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


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
        # words = words.union(set([line.strip().split()[0]  for line in open(file, "r", encoding='utf-8').readlines()]))
        # tags = tags.union(set([line.strip().split()[-1] for line in open(file, "r", encoding='utf-8').readlines()]))
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


def tokenize(filename, vocab2id, tag2id, maxlen):
    """
    給定 train.data 輸出 tokenize 過的文字列表

    Args:
        filename:
            train.data
            中 O
            文 O

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
    contents = []
    labels = []
    content = []
    label = []

    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            try:
                if line != "\n":
                    w, t = line.split()
                    content.append(vocab2id.get(w, 0))
                    label.append(tag2id.get(t, 0))
                else:
                    if content and label:
                        contents.append(content)
                        labels.append(label)
                    content = []
                    label = []
            except Exception as e:
                content = []
                label = []
    contents = tf.keras.preprocessing.sequence.pad_sequences(
        contents, padding="post", maxlen=maxlen
    )
    labels = tf.keras.preprocessing.sequence.pad_sequences(
        labels, padding="post", maxlen=maxlen
    )
    return contents, labels


tag_check = {
    "I": ["B", "I"],
    "E": ["B", "I"],
}


def check_label(front_label, follow_label):
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (
        (follow_label.startswith("I-") or follow_label.startswith("E-"))
        and front_label.endswith(follow_label.split("-")[1])
        and front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]
    ):
        return True
    return False


def format_result(chars, tags):
    """
    將 TEXT 和 TAG 抓出來，回傳 entity 列表。

    Args:
        chars: ['国','家','发','展','计','划','委','员','会','副','主','任','王','春','正']

        tags: ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'E-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'E-PER']

    Returns:
        [{'begin': 0, 'end': 9, 'words': '国家发展计划委员会', 'type': 'ORG'},
         {'begin': 12, 'end': 15, 'words': '王春正', 'type': 'PER'}]
    """

    entities = []
    entity = []
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])
    if entity:
        entities.append(entity)

    entities_result = []
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {
                    "begin": entity[0][0],
                    "end": entity[-1][0] + 1,
                    "words": "".join([char for _, char, _, _ in entity]),
                    "type": entity[0][2].split("-")[1],
                }
            )

    return entities_result


# %%
class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size):
        super(NerModel, self).__init__()
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_num, return_sequences=True)
        )
        self.dense = tf.keras.layers.Dense(label_size)

        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(label_size, label_size))
        )
        self.dropout = tf.keras.layers.Dropout(0.5)

    @tf.function
    def call(self, text, labels=None, training=None):
        text_lens = tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1
        )
        # -1 change 0
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
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


# %%
vocab_file_path = "../../dataset/crf_bilstm/vocab_file.txt"
tag_file_path = "../../dataset/crf_bilstm/tag.txt"

train_path = "../../dataset/crf_data/train_grained.data"

if not (os.path.exists(vocab_file_path) and os.path.exists(tag_file_path)):
    build_vocab([train_path], vocab_file_path, tag_file_path)

BATCH_SIZE = 128
HIIDEN_NUMS = 512
EMBEDDING_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 20
SENTENCE_MAX_LEN = 32
checkpoint_file_path = "../checkpoints/crf_bilstm/"


# %%
voc2id, id2voc = read_vocab(vocab_file_path)
tag2id, id2tag = read_vocab(tag_file_path)

text_sequences, label_sequences = tokenize(train_path, voc2id, tag2id, SENTENCE_MAX_LEN)


#%%
train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, label_sequences))
train_dataset = train_dataset.shuffle(len(text_sequences)).batch(
    BATCH_SIZE, drop_remainder=True
)


model = NerModel(
    hidden_num=HIIDEN_NUMS,
    vocab_size=len(voc2id),
    label_size=len(tag2id),
    embedding_size=EMBEDDING_SIZE,
)

optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_file_path))
ckpt_manager = tf.train.CheckpointManager(
    ckpt, checkpoint_file_path, checkpoint_name="model.ckpt", max_to_keep=3
)


# %%
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


#%%
best_acc = 0
step = 0

for epoch in range(EPOCHS):
    for _, (text_batch, labels_batch) in enumerate(train_dataset):
        step = step + 1
        loss, logits, text_lens = train_one_step(text_batch, labels_batch)
        if step % 20 == 0:
            accuracy = get_acc_one_step(logits, text_lens, labels_batch)
            print(
                f"epoch {epoch}, step {step}, loss {loss:.4f} , accuracy {accuracy:.4f}"
            )

            if accuracy > best_acc:
                best_acc = accuracy
                ckpt_manager.save()
                print("model saved")

# %%
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
model = NerModel(
    hidden_num=HIIDEN_NUMS,
    vocab_size=len(voc2id),
    label_size=len(tag2id),
    embedding_size=EMBEDDING_SIZE,
)


# restore model
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_file_path))


#%%
# predict single sentence
def predict_sentence(sentence):
    # dataset = encode sentence
    #         = [[1445   33 1878  826 1949 1510  112]]
    dataset = tf.keras.preprocessing.sequence.pad_sequences(
        [[voc2id.get(char, 0) for char in sentence]], padding="post"
    )

    # logits = (1, 7, 28) = (sentence, words, predict_distrib)
    # text_lens = [7]
    logits, text_lens = model.predict(dataset)

    paths = []

    for logit, text_len in zip(logits, text_lens):
        viterbi_path, _ = tf_ad.text.viterbi_decode(
            logit[:text_len], model.transition_params
        )
        paths.append(viterbi_path)

    # path[0] = tag in sentence
    #         = [18, 19, 19, 1, 26, 27, 1]

    # result  = ['B-name', 'I-name', 'I-name', 'O', 'B-time', 'I-time', 'O']
    result = [id2tag[id] for id in paths[0]]

    # entities_result =
    # [{'begin': 0, 'end': 3, 'words': '賈伯斯', 'type': 'name'},
    #  {'begin': 4, 'end': 6, 'words': '七號', 'type': 'time'}]
    entities_result = format_result(list(sentence), result)
    return entities_result


test_path = "../../dataset/crf_data/test_grained.data"
test_raw_path = "../../dataset/test.txt"
test_mapping = [len(article) for article in loadTestFile(test_raw_path)]

# predict testset
def predict(test_mapping, test_data_path):
    """
    test_mapping:
        test.txt 每一篇的長度

    test_data_path:
        test_grained.data 的路徑
    """
    testsets = []
    content = []
    with open(test_data_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            if line != "\n":
                w = line.strip("\n")
                content.append(w)
            else:
                if content:
                    testsets.append("".join(content))
                content = []

    article_id = 0
    counter = 0
    results = []
    result = []
    for testset in tqdm(testsets):
        prediction = predict_sentence(testset)

        # predict_pos + counter
        if prediction:
            for pred in prediction:
                pred["begin"] += counter
                pred["end"] += counter
                result.append(pred)

        counter += len(testset)

        if counter == test_mapping[article_id]:
            results.append(result)
            article_id += 1
            counter = 0
            result = []

    return results


results = predict(test_mapping, test_path)

#%%   

output_path = "../../output/output.tsv"

def output_result_tsv(results):
    """
    將 results 輸出成 tsv

    results list (article, results, result-tuple):
        [
            [
                {'begin': 170, 'end': 174, 'words': '1100', 'type': 'med_exam'},
                {'begin': 245, 'end': 249, 'words': '1145', 'type': 'med_exam'},
                ...
            ]
            ,
            [
                {'begin': 11, 'end': 13, 'words': '一天', 'type': 'time'},
                {'begin': 263, 'end': 267, 'words': '今天早上', 'type': 'time'},
                ...
            ]
        ]

    """
    titles = {"end": "end_position", "begin": "start_position", "words": "entity_text", "type": "entity_type"}
    df = pd.DataFrame()
    
    for i, result in enumerate(results):
        results = pd.DataFrame(result).rename(columns=titles)
        results = results[['start_position', 'end_position', 'entity_text', 'entity_type']]
        
        article_ids = pd.Series([i]*len(result), name="article_id")
        df = df.append(pd.concat([article_ids, results], axis=1),  ignore_index=True)

    df.to_csv(output_path, sep="\t", index=False)
    
output_result_tsv(results)
# %%
