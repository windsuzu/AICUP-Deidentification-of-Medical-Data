# %%
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.append(str(Path().resolve().parents[1]))
import program.deep_learning.custom_metrics as custom_metrics

MAX_SENTENCE_LENGTH = 128
BERT_TOKENS_COUNT = 2


def getTrainData(root_datapath):

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
    path = Path().resolve().parents[1] / root_datapath
    with path.open(encoding="UTF8") as f:
        train_data = f.readlines()
        texts = []
        tags = []

        # content of each line: "中 B-name\n"
        # line[0] = 中
        # line[2:-1] = B-name
        for line in train_data:
            if line == "\n" or len(texts) == MAX_SENTENCE_LENGTH:
                train_texts.append(texts)
                train_tags.append(tags)
                texts = []
                tags = []

            if line == "\n":
                continue

            text, tag = line[0], line[2:-1]
            texts.append(text)
            tags.append(tag)

    return train_texts, train_tags


def getTestData(root_datapath):
    """Load crf's test.data to get the test data in the format of the transformers.

    Returns:
        test_texts (dataset_size, content_size):
            [...[...'有', '辦', '法', '，', '這', '是', '嚴', '重', '或', '一', ...]...]

        test_mapping: test texts without split.
    """
    test_texts = []
    test_mapping = []
    path = Path().resolve().parents[1] / root_datapath
    with path.open(encoding="UTF8") as f:
        test_data = f.readlines()
        texts = []
        mappings = []

        for line in test_data:
            if line == "\n" or len(texts) == MAX_SENTENCE_LENGTH:
                test_texts.append(texts)
                mappings.extend(texts)
                texts = []

            if line == "\n":
                test_mapping.append(mappings)
                mappings = []
                continue

            text = line[0]
            texts.append(text)

    return test_texts, test_mapping


def encode_tags(tags):
    unaligned_labels = [[tag2id[tag] for tag in content] for content in tags]
    token_o = tag2id["O"]
    bert_token_unaligned_labels = map(
        lambda labels: [token_o, *labels, token_o], unaligned_labels
    )

    return tf.keras.preprocessing.sequence.pad_sequences(
        list(bert_token_unaligned_labels),
        value=-1,
        padding="post",
        truncating="post",
        maxlen=MAX_SENTENCE_LENGTH + BERT_TOKENS_COUNT,
    )


# %%
train_texts, train_tags = getTrainData("dataset/crf_data/train.data")
test_texts, test_mapping = getTestData("dataset/crf_data/test.data")

unique_tags = set(tag for tags in train_tags for tag in tags)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
num_labels = len(unique_tags)

# %%
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

train_texts, val_texts, train_tags, val_tags = train_test_split(
    train_texts, train_tags, test_size=0.1
)

tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")

# %%
train_encodings = tokenizer(
    train_texts,
    is_split_into_words=True,
    padding=True,
    truncation=True,
    return_token_type_ids=False,
)
val_encodings = tokenizer(
    val_texts,
    is_split_into_words=True,
    padding=True,
    truncation=True,
    return_token_type_ids=False,
)
test_encodings = tokenizer(
    test_texts,
    is_split_into_words=True,
    padding=True,
    truncation=True,
    return_token_type_ids=False,
)

train_labels = encode_tags(train_tags)
val_labels = encode_tags(val_tags)

# [CLS] + sequence + [SEP] = 128 + 2
# train_encoding, attention_mask, label:
# [CLS] (101, 1, 10)
# 好  (1962, 1, 10)
# 喔  (1595, 1, 10)
# ...
# [SEP] (102, 1, 10)
# [PAD] (0, 0, -1)
# [PAD] (0, 0, -1)
# ...


# %%
train_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(train_encodings), train_labels)
)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings)))

# %%
from transformers import TFAutoModelForTokenClassification

model = TFAutoModelForTokenClassification.from_pretrained(
    "ckiplab/bert-base-chinese-ner", from_pt=True, output_hidden_states=True
)

from tf2crf import CRF, ModelWithCRFLoss

input_shape = MAX_SENTENCE_LENGTH + BERT_TOKENS_COUNT

input_ids = tf.keras.Input(name="input_ids", shape=(input_shape,), dtype=tf.int32)
attention_mask = tf.keras.Input(
    name="attention_mask", shape=(input_shape,), dtype=tf.int32
)

transformer = model([input_ids, attention_mask])
hidden_states = transformer[1]

hidden_states_size = 1
hiddes_states_ind = list(range(-hidden_states_size, 0, 1))

selected_hiddes_states = tf.keras.layers.concatenate(
    tuple([hidden_states[i] for i in hiddes_states_ind])
)

# crf = CRF(num_labels, name="crf_layer")
crf = CRF(dtype='float32')

output = tf.keras.layers.Dense(num_labels, activation="relu")(selected_hiddes_states)
output = crf(output)

model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
model = ModelWithCRFLoss(model)
# model.summary()


#%%
yp = None
yt = None


##%%
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "../checkpoints/crf_transformers/",
    save_best_only=True,
    save_weights_only=True,
    moniter="val_f1",
    mode="max",
)


##%%
import matplotlib.pyplot as plt

history = model.fit(
    train_dataset.shuffle(1000).batch(4),
    epochs=5,
    batch_size=4,
    validation_data=val_dataset.batch(4),
    callbacks=[custom_metrics.Metrics(val_dataset), checkpoint],
)

# %%
tf.keras.models.save_model(model, "../checkpoints/crf_transformers/1")
plt.plot(history.history["val_f1"])
plt.title("Model f1")
plt.ylabel("f1")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()


# %%
prediction = model.predict(test_dataset)[0]
# prediction = np.argmax(prediction, -1)
# batch_size = MAX_SENTENCE_LENGTH + BERT_TOKENS_COUNT
# prediction = prediction.reshape(-1, batch_size)


# %%
# output_format:
# (article_id, start_position, end_position, entity_text, entity_type)

output = []

article_id = 0
start_batch = 0
end_batch = 0

for article in test_mapping:
    start_batch = end_batch
    end_batch += (len(article) // MAX_SENTENCE_LENGTH) + 1

    pos_counter = 0
    entity_type = None
    start_pos = None
    end_pos = None

    for preds in prediction[start_batch:end_batch]:
        # get rid of [CLS], [SEP] in common batches
        # exceptions only occur in last batches, no matters
        preds = preds[1:-1]

        for i, pred in enumerate(preds):
            if id2tag[pred][0] == "B":
                start_pos = pos_counter
                entity_type = id2tag[pred][2:]  # remove "B-"
            elif id2tag[pred][0] == "I":
                end_pos = pos_counter
            elif id2tag[pred][0] == "O" or i + 1 == MAX_SENTENCE_LENGTH:
                if entity_type:
                    entity_name = article[start_pos : (end_pos + 1)]
                    output.append(
                        (article_id, start_pos, end_pos, entity_name, entity_type)
                    )
                    entity_type = None
            pos_counter += 1
