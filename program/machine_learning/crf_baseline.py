# %% import block
import os
import numpy as np
import sys

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from pathlib import Path

sys.path.append(str(Path().resolve().parents[1]))
from program.utils.load_data import loadTestFile
from program.utils.write_output_file import generateOutputFile

validate = False
test_data_delete_blank = True

pretrained_word2vec_path = "data/cna.cbow.cwe_p.tar_g.512d.0.txt"
output_path = "output/output.tsv"

train_data_path = "../../dataset/crf_data/train.data"
test_data_path = "../../dataset/crf_data/test.data"
test_file_path = "../../dataset/test.txt"

testingset = loadTestFile(test_file_path)

# %% CRF Model
model = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    verbose=True,
    all_possible_transitions=True,
)

# %% load pretrained word vectors
# get a dict of tokens (key) and their pretrained word vectors (value)
# pretrained word2vec CBOW word vector:
# https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305ddf5522015de5479f4701b1
word_vecs = {}
with open(pretrained_word2vec_path, "r", encoding="utf-8") as word2vec:
    for line in word2vec:
        tokens = line.strip().split()

        # There are two integers in the first line: vocabulary_size, word_vector_dim
        if len(tokens) == 2:
            continue

        word = tokens[0]
        vec = np.array([float(dim) for dim in tokens[1:]])
        word_vecs[word] = vec

print("vocabulary_size: ", len(word_vecs))
print("word_vector_dim: ", vec.shape)


# %% Fit required data sturture of crf model
def CRFFormatToList(data_path):

    with open(data_path, "r", encoding="utf-8") as format_data:
        data = format_data.readlines()  # .encode('utf-8').decode('utf-8-sig')

    state = "test" if len(data[0].strip("\n").split(" ")) == 1 else "train"
    total_data_list, separate_data_list = list(), list()
    for row in data:
        data_tuple = tuple()
        if row == "\n":
            total_data_list.append(separate_data_list)
            separate_data_list = []
        else:
            if state == "test":
                token = row.strip("\n").split(" ")
                separate_data_list.append(token)
            else:
                token, label = row.strip("\n").split(" ")
                token_label_pair = (token, label)
                separate_data_list.append(token_label_pair)

    return total_data_list


def GetUntokenVector(word_vector_dim):
    np.random.seed(42)
    unk_vector = np.random.rand(word_vector_dim)
    return unk_vector


def ListToDict(vector):
    vector_dict = dict()
    for idx_vec in range(len(vector)):
        vector_dict["dim_" + str(idx_vec + 1)] = vector[idx_vec]
    return vector_dict


def Word2Vector(separate_data_list, embedded_dict, unk_vector):
    separate_embedded_list = list()
    for idx_tuple in range(len(separate_data_list)):
        token = separate_data_list[idx_tuple][0]

        if token in embedded_dict:
            vector = embedded_dict[token]
        else:
            vector = unk_vector
        vector = ListToDict(vector)
        separate_embedded_list.append(vector)

    return separate_embedded_list


def GetInputData(data_list, embedded_dict):
    input_list = list()

    # No Match Word (unknown word) Vector in Embedding
    word_vector_dim = len(list(embedded_dict.values())[0])
    unk_vector = GetUntokenVector(word_vector_dim)

    for idx_list in range(len(data_list)):
        separate_separate_list = Word2Vector(
            data_list[idx_list], embedded_dict, unk_vector
        )
        input_list.append(separate_separate_list)

    return input_list


def GetLabelData(data_list):
    total_label_list = list()
    for idx_list in range(len(data_list)):
        separate_label_list = list()
        for idx_tuple in range(len(data_list[idx_list])):
            separate_label_list.append(data_list[idx_list][idx_tuple][1])
        total_label_list.append(separate_label_list)
    return total_label_list


# %%
train_data_list = CRFFormatToList(train_data_path)
test_data_list = CRFFormatToList(test_data_path)

# %%
if validate:
    print("Processing validate data")
    train_data_list, val_data_list = train_test_split(
        train_data_list, test_size=0.25, random_state=42
    )
    x_val = GetInputData(val_data_list, word_vecs)
    y_val = GetLabelData(val_data_list)

print("Processing training data")
x_train = GetInputData(train_data_list, word_vecs)
y_train = GetLabelData(train_data_list)

print("Processing testing data")
x_test = GetInputData(test_data_list, word_vecs)

print("Training...")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# %%
def ModelEvaluation(model, ground_truth, predict):
    labels = list(model.classes_)
    labels.remove("O")
    f1_score = metrics.flat_f1_score(
        ground_truth, predict, average="weighted", labels=labels
    )
    sorted_labels = sorted(
        labels, key=lambda name: (name[1:], name[0])
    )  # group B and I results

    print(
        flat_classification_report(
            ground_truth, predict, labels=sorted_labels, digits=3
        )
    )
    print("F1 score :", f1_score)


if validate:
    y_pred_val = model.predict(x_val)
    ModelEvaluation(model, y_val, y_pred_val)

# %% Output in upload format
generateOutputFile(
    y_pred, test_data_list, testingset, output_path, test_data_delete_blank
)
