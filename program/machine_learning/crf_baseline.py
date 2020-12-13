# %% import block
import sys
import sklearn_crfsuite
import numpy as np
from tqdm import tqdm, trange
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from program.utils.crf_preprocessing import loadInputFile, loadTestFile, CRFFormatData

train_file_path = "../../dataset/train.txt"
test_file_path = "../../dataset/development.txt"

# %% CRF Model
def CRF(x_train, y_train, x_test):
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        verbose=True,
        all_possible_transitions=True,
    )
    crf.fit(x_train, y_train)
    y_test_pred = crf.predict(x_test)
    # y_val_pred_mar = crf.predict_marginals(x_val)

    # labels = list(crf.classes_)
    # labels.remove('O')

    # val_f1score = metrics.flat_f1_score(y_val,
    #                                 y_val_pred,
    #                                 average='weighted',
    #                                 labels=labels)
    # sorted_labels = sorted(labels, key=lambda name:
    #                        (name[1:], name[0]))  # group B and I results
    # print(
    #     flat_classification_report(y_val,
    #                                y_val_pred,
    #                                labels=sorted_labels,
    #                                digits=3))

    return y_test_pred


# %% Data preprocess method
def Dataset(data_path):
    """
    load `train.data` and separate into a list of labeled data of each text

    return:
        data_list:
        a list of lists of tuples, storing tokens and labels (wrapped in tuple) of each text in `train.data`

        traindata_list:
        a list of lists, storing training data_list splitted from data_list

        testdata_list:
        a list of lists, storing testing data_list splitted from data_list
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()  # .encode('utf-8').decode('utf-8-sig')
    data_list, data_list_tmp = list(), list()
    article_id_list = list()
    idx = 0
    for row in data:
        data_tuple = tuple()
        if row == "\n":
            article_id_list.append(idx)
            idx += 1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip("\n").split(" ")
            if len(row) == 1:
                data_tuple = row[0]
            else:
                data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)

    # here we random split data into training dataset and testing dataset
    # but you should take `development data` or `test data` as testing data
    # At that time, you could just delete this line,
    # and generate data_list of `train data` and data_list of `development/test data` by this function

    # traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = train_test_split(data_list, article_id_list, test_size=0.05, random_state=42)
    # return data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list

    return data_list, article_id_list


# look up word vectors
# turn each word into its pretrained word vector
# return a list of word vectors corresponding to each token in train.data
def Word2Vector(data_list, embedding_dict):
    embedding_list = list()

    # No Match Word (unknown word) Vector in Embedding
    unk_vector = np.random.rand(*(list(embedding_dict.values())[0].shape))

    for idx_list in trange(len(data_list)):
        embedding_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            key = data_list[idx_list][idx_tuple][0]  # token

            if key in embedding_dict:
                value = embedding_dict[key]
            else:
                value = unk_vector
            embedding_list_tmp.append(value)
        embedding_list.append(embedding_list_tmp)
    return embedding_list


def Feature(embed_list):
    """
    # input features: pretrained word vectors of each token
    # return a list of feature dicts, each feature dict corresponding to each token
    """
    feature_list = list()
    for idx_list in trange(len(embed_list)):  # for every article
        feature_list_tmp = list()
        # for every word (512)
        for idx_tuple in range(len(embed_list[idx_list])):
            feature_dict = dict()
            for idx_vec in range(len(embed_list[idx_list][idx_tuple])):
                feature_dict["dim_" + str(idx_vec + 1)] = embed_list[idx_list][
                    idx_tuple
                ][idx_vec]
            feature_list_tmp.append(feature_dict)
        feature_list.append(feature_list_tmp)
    return feature_list


def Preprocess(data_list):
    """
    # get the labels of each tokens in train.data
    # return a list of lists of labels
    """
    label_list = list()
    for idx_list in range(len(data_list)):
        label_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            label_list_tmp.append(data_list[idx_list][idx_tuple][1])
        label_list.append(label_list_tmp)
    return label_list


# %%
trainingset, position, mentions = loadInputFile(train_file_path)
testset = loadTestFile(test_file_path)

train_data_path = "../../dataset/crf_data/train.data"
CRFFormatData(trainingset, train_data_path, position)

test_data_path = "../../dataset/crf_data/test.data"
CRFFormatData(testset, test_data_path)

# %% load pretrained word vectors
# get a dict of tokens (key) and their pretrained word vectors (value)
# pretrained word2vec CBOW word vector:
# https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305ddf5522015de5479f4701b1
dim = 0
word_vecs = {}
# open pretrained word vector file
with open("../../pretrained/cna.cbow.cwe_p.tar_g.512d.0.txt", encoding="utf8") as f:
    for line in tqdm(f):
        tokens = line.strip().split()

        if len(tokens) == 2:
            dim = int(tokens[1])
            continue

        word = tokens[0]
        vec = np.array([float(t) for t in tokens[1:]])
        word_vecs[word] = vec

print("vocabulary_size: ", len(word_vecs))
print("word_vector_dim: ", vec.shape)

# %%
train_data_list, train_data_article_id_list = Dataset(train_data_path)
test_data_list, test_data_article_id_list = Dataset(test_data_path)

# %%
# Load Word Embedding
train_embed_list = Word2Vector(train_data_list, word_vecs)
test_embed_list = Word2Vector(test_data_list, word_vecs)

# CRF - Train Data (Augmentation Data)
x_train = Feature(train_embed_list)
y_train = Preprocess(train_data_list)

# CRF - Test Data (Golden Standard)
x_test = Feature(test_embed_list)
# y_val = Preprocess(val_data_list)

# %% Training & Predicting
y_pred = CRF(x_train, y_train, x_test)

# %% Output in upload format

output = "article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
for test_id in range(len(y_pred)):  # for every content
    pos = 0
    start_pos = None
    end_pos = None
    entity_text = None
    entity_type = None
    # for every predict token in the content
    for pred_id in range(len(y_pred[test_id])):
        if y_pred[test_id][pred_id][0] == "B":
            start_pos = pos
            entity_type = y_pred[test_id][pred_id][2:]
        elif (
            start_pos is not None
            and y_pred[test_id][pred_id][0] == "I"
            and (
                pred_id + 1 == len(y_pred[test_id])
                or y_pred[test_id][pred_id + 1][0] == "O"
            )
        ):
            end_pos = pos
            entity_text = "".join(
                [
                    test_data_list[test_id][position][0]
                    for position in range(start_pos, end_pos + 1)
                ]
            )
            line = (
                str(test_data_article_id_list[test_id])
                + "\t"
                + str(start_pos)
                + "\t"
                + str(end_pos + 1)
                + "\t"
                + entity_text
                + "\t"
                + entity_type
            )
            output += line + "\n"
        pos += 1

output_path = "../output/output.tsv"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(output)

#%%

# You may try python-crfsuite to train an neural network for NER tagging optimized by gradient descent back propagation

# You may try CRF++ tool for NER tagging by CRF model

# You may try other traditional chinese word embedding (ex. fasttext, bert, ...) for input features

# You may try add other features for NER model, ex. POS-tag, word_length, word_position, ...

# You should upload the prediction output on development data or test data provided later to the # competition system.

# Note don't upload prediction output on the splitted testing dataset like this # baseline example.