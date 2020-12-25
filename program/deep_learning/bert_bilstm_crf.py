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

MAX_SENTENCE_LENGTH = 32
BATCH_SIZE = 16
HIIDEN_NUMS = 512
LEARNING_RATE = 1e-3
EPOCHS = 15
checkpoint_file_path = "../checkpoints/bert_bilstm_crf/"


def getTrainData(datapath):

    """
    Load crf's train.data to get the train data in the format of the transformers.

    Returns:
        train_texts (dataset_size, content_size):
            [...[...'：', '阿', '只', '是', '前', '天', '好', '很', '多'...]...]

        train_tags (dataset_size, content_size):
            [...[...'O', 'O', 'O', 'O', 'B-time', 'I-time', 'O', 'O', 'O'...]...]
    """
    train_texts = []
    train_tags = []
    path = Path().resolve().parents[1] / datapath
    with path.open(encoding="UTF8") as f:
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


def encode_tags(tags):
    return tf.keras.preprocessing.sequence.pad_sequences(
        [[tag2id[label] for label in tag] for tag in tags],
        value=tag2id["O"],
        padding="post",
        truncating="post",
        maxlen=MAX_SENTENCE_LENGTH,
    )


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


#%%
from transformers import BertTokenizer, TFBertModel

bert_model_name = [
    "hfl/chinese-bert-wwm",
    "hfl/chinese-bert-wwm-ext",
    "hfl/chinese-roberta-wwm-ext",
    "chinese-roberta-wwm-ext-large",
]

tokenizer = BertTokenizer.from_pretrained(bert_model_name[3])
bert_model = TFBertModel.from_pretrained(bert_model_name[3], from_pt=True)


#%%
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


train_texts, train_tags = getTrainData("/gdrive/My Drive/AICUP2020/dataset/crf_data/train_grained.data")
# train_texts, train_tags = remove_imbalance_trainsets(train_texts, train_tags)
unique_tags = set(tag for tags in train_tags for tag in tags)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


#%%
train_encodings = tokenizer(
    train_texts,
    add_special_tokens=False,
    return_token_type_ids=False,
    is_split_into_words=True,
    padding="max_length",
    truncation=True,
    max_length=MAX_SENTENCE_LENGTH,
)

train_labels = encode_tags(train_tags)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(train_encodings), train_labels)
)
train_dataset = train_dataset.shuffle(len(train_texts)).batch(
    BATCH_SIZE, drop_remainder=True
)


#%%
class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, label_size):
        super(NerModel, self).__init__()

        self.embedding = bert_model

        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_num, return_sequences=True)
        )
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(label_size)

        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(label_size, label_size))
        )

    @tf.function
    def call(self, text, labels=None, training=None):
        text_lens = tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(text["input_ids"], 0), dtype=tf.int32), axis=-1
        )
        # -1 change 0
        # 64, 32, 768
        embedding_layer = self.embedding(
            text["input_ids"], attention_mask=text["attention_mask"]
        )[0]
        
        inputs = self.dropout(embedding_layer, training)
        # 64, 32, 27
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
model = NerModel(
    hidden_num=HIIDEN_NUMS,
    label_size=len(tag2id),
)

# freeze BERT embedding parameters
model.layers[0].trainable = False

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
        step += 1
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


#%%
# restore model
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_file_path))

# predict single sentence
def predict_sentence(sentence):
    # dataset = encode sentence
    #         = [[1445   33 1878  826 1949 1510  112]]
    dataset = tokenizer(sentence, add_special_tokens=False, return_token_type_ids=False, is_split_into_words=True, padding=True)
    dataset = tf.data.Dataset.from_tensors(dict(dataset)).batch(1)

    # logits = (1, 7, 28) = (sentence, words, predict_distrib)
    # text_lens = [7]
    logits, text_lens = model.predict(dataset)
    paths = []
    logits = logits.squeeze()[np.newaxis, :]
    text_lens = [sum(text_lens)]

    for logit, text_len in zip(logits, text_lens):
        viterbi_path, _ = tf_ad.text.viterbi_decode(
            logit[:text_len], model.transition_params
        )

        paths.append(viterbi_path)

    # paths[0] = tag in sentence
    #          = [18, 19, 19, 1, 26, 27, 1]

    # result  = ['B-name', 'I-name', 'I-name', 'O', 'B-time', 'I-time', 'O']
    result = [id2tag[id] for id in paths[0]]
    # entities_result =
    # [{'begin': 0, 'end': 3, 'words': '賈伯斯', 'type': 'name'},
    #  {'begin': 4, 'end': 6, 'words': '七號', 'type': 'time'}]
    entities_result = format_result(list(sentence), result)
    return entities_result


# predict_sentence("醫生：賈伯斯是七號。")
# predict_sentence("民眾：阿只是前天好很多。前天就算沒盜，可是一覺到天明這樣。")
# predict_sentence("民眾：嗯。")


#%%
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
    titles = {
        "end": "end_position",
        "begin": "start_position",
        "words": "entity_text",
        "type": "entity_type",
    }
    df = pd.DataFrame()

    for i, result in enumerate(results):
        if not result:
            continue
        results = pd.DataFrame(result).rename(columns=titles)
        results = results[
            ["start_position", "end_position", "entity_text", "entity_type"]
        ]
        article_ids = pd.Series([i] * len(result), name="article_id")
        df = df.append(pd.concat([article_ids, results], axis=1), ignore_index=True)

    df.to_csv(output_path, sep="\t", index=False)


output_result_tsv(results)