#%%
import sys
from pathlib import Path
from tqdm import tqdm

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