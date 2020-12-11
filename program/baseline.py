# %% import block
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
import sklearn_crfsuite
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

train_file_path = '../data/train_0.txt'
test_file_path = '../dataset/development_1.txt'


# %% function block
def loadInputFile(path):
    # store trainingset [content,content,...]
    trainingset = list()
    # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    position = list()    
    # store mentions [entity_text] = entity_type                       
    mentions = dict()

    with open(path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]

    for data in datas:
        data = data.split('\n')
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]

        for annot in annotations[1:]:
            # annot = article_id, start_pos, end_pos, entity_text, entity_type
            annot = annot.split('\t')
            position.extend(annot)
            mentions[annot[3]] = annot[4]

    return trainingset, position, mentions

def loadTestFile(path):
    testset = list()  # store testset [content,content,...]
    with open(path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
        datas = file_text.split('\n\n--------------------\n\n')[:-1]
        for data in datas:
            data = data.split('\n')
            testset.append(data[1])

    return testset

# def RemoveBlankSpace(data):
#     while '' or ' ' in data:
#         if '' in data:
#             data.remove('')
#         else:
#             data.remove(' ')
#     return data

def GenerateFormatData(dataset, path, position=0):
    
    if (os.path.isfile(path)):
        print("Have been generated")
        return

    outputfile = open(path, 'w', encoding= 'utf-8')
    state = "train" if position else "test"

    if state == "test":
        for article_id in range(len(dataset)):
#             testset_split = list(dataset[article_id])
#             clear_trainingset = RemoveBlankSpace(testset_split)
            
            content = "\n".join([word for word in dataset[article_id]])
            outputfile.write(content)
            outputfile.write("\n\n")

            if article_id % 10 == 0:
                print('Total complete articles:', article_id)
    else:
        count = 0  # annotation counts in each content
        tagged = list()
        for article_id in range(len(dataset)):
    #         trainingset_split = list(dataset[article_id])
    #         clear_trainingset = RemoveBlankSpace(trainingset_split)           ### 根本沒用到，後面還做了一次= =

            start_tmp = 0
            for position_idx in range(0, len(position), 5):                     ### 這是三小啦？？
                if int(position[position_idx]) == article_id:                   ### 做了相當多不必要的回圈 可以改成while
                    count += 1
                    if count == 1:                                              ### 如果不是第1個，前面要補0
                        start_pos = int(position[position_idx + 1])
                        end_pos = int(position[position_idx + 2])
                        entity_type = position[position_idx + 4]
                        if start_pos == 0:
                            token = list(dataset[article_id][start_pos:end_pos])
                            whole_token = dataset[article_id][start_pos:end_pos]
                            for token_idx in range(len(token)):
                                if len(token[token_idx].replace(' ', '')) == 0: ### 很棒 他媽都沒有空白格
                                    continue
                                # BIO states
                                if token_idx == 0:
                                    label = 'B-' + entity_type
                                else:
                                    label = 'I-' + entity_type

                                output_str = token[token_idx] + ' ' + label + '\n'
                                outputfile.write(output_str)

                        else:
                            token = list(dataset[article_id][0:start_pos])
                            whole_token = dataset[article_id][0:start_pos]
                            for token_idx in range(len(token)):
                                if len(token[token_idx].replace(' ', '')) == 0:
                                    continue

                                output_str = token[token_idx] + ' ' + 'O' + '\n'
                                outputfile.write(output_str)

                            token = list(dataset[article_id][start_pos:end_pos])
                            whole_token = dataset[article_id][start_pos:end_pos]
                            for token_idx in range(len(token)):
                                if len(token[token_idx].replace(' ', '')) == 0:
                                    continue
                                # BIO states
                                if token[0] == '':
                                    if token_idx == 1:
                                        label = 'B-' + entity_type
                                    else:
                                        label = 'I-' + entity_type
                                else:
                                    if token_idx == 0:
                                        label = 'B-' + entity_type
                                    else:
                                        label = 'I-' + entity_type

                                output_str = token[token_idx] + ' ' + label + '\n'
                                outputfile.write(output_str)

                        start_tmp = end_pos
                    else:
                        start_pos = int(position[position_idx + 1])
                        end_pos = int(position[position_idx + 2])
                        entity_type = position[position_idx + 4]
                        if start_pos < start_tmp:
                            continue
                        else:
                            token = list(dataset[article_id][start_tmp:start_pos])
                            whole_token = dataset[article_id][start_tmp:start_pos]
                            for token_idx in range(len(token)):
                                if len(token[token_idx].replace(' ', '')) == 0:
                                    continue
                                output_str = token[token_idx] + ' ' + 'O' + '\n'
                                outputfile.write(output_str)

                        token = list(dataset[article_id][start_pos:end_pos])
                        whole_token = dataset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token[0] == '':
                                if token_idx == 1:
                                    label = 'B-' + entity_type
                                else:
                                    label = 'I-' + entity_type
                            else:
                                if token_idx == 0:
                                    label = 'B-' + entity_type
                                else:
                                    label = 'I-' + entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)
                        start_tmp = end_pos

            token = list(dataset[article_id][start_tmp:])
            whole_token = dataset[article_id][start_tmp:]
            for token_idx in range(len(token)):
                if len(token[token_idx].replace(' ', '')) == 0:
                    continue

                output_str = token[token_idx] + ' ' + 'O' + '\n'
                outputfile.write(output_str)

            count = 0

            output_str = '\n'
            outputfile.write(output_str)
            ID = dataset[article_id]

            if article_id % 10 == 0:
                print('Total complete articles:', article_id)

    # close output file
    outputfile.close()


# %% CRF Model
def CRF(x_train, y_train, x_test):
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                               c1=0.1,
                               c2=0.1,
                               max_iterations=100,
                               verbose=True,
                               all_possible_transitions=True)
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
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # .encode('utf-8').decode('utf-8-sig')
    data_list, data_list_tmp = list(), list()
    article_id_list = list()
    idx = 0
    for row in data:
        data_tuple = tuple()
        if row == '\n':
            article_id_list.append(idx)
            idx += 1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
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
                feature_dict['dim_' +
                             str(idx_vec +
                                 1)] = embed_list[idx_list][idx_tuple][idx_vec]
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

train_data_path = '../dataset/crf_data/train.data'
GenerateFormatData(trainingset, train_data_path, position)

test_data_path = '../dataset/crf_data/test.data'
GenerateFormatData(testset, test_data_path)

# %% load pretrained word vectors
# get a dict of tokens (key) and their pretrained word vectors (value)
# pretrained word2vec CBOW word vector:
# https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305ddf5522015de5479f4701b1
dim = 0
word_vecs = {}
# open pretrained word vector file
with open('../pretrained/cna.cbow.cwe_p.tar_g.512d.0.txt',
          encoding='utf8') as f:
    for line in tqdm(f):
        tokens = line.strip().split()

        if len(tokens) == 2:
            dim = int(tokens[1])
            continue

        word = tokens[0]
        vec = np.array([float(t) for t in tokens[1:]])
        word_vecs[word] = vec

print('vocabulary_size: ', len(word_vecs))
print('word_vector_dim: ', vec.shape)

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
        if y_pred[test_id][pred_id][0] == 'B':
            start_pos = pos
            entity_type = y_pred[test_id][pred_id][2:]
        elif start_pos is not None and y_pred[test_id][pred_id][0] == 'I' and (
                pred_id + 1 == len(y_pred[test_id])
                or y_pred[test_id][pred_id + 1][0] == 'O'):
            end_pos = pos
            entity_text = ''.join([
                test_data_list[test_id][position][0]
                for position in range(start_pos, end_pos + 1)
            ])
            line = str(test_data_article_id_list[test_id]) + '\t' + str(
                start_pos) + '\t' + str(
                    end_pos + 1) + '\t' + entity_text + '\t' + entity_type
            output += line + '\n'
        pos += 1

output_path = '../output/output.tsv'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(output)

#%%

# You may try python-crfsuite to train an neural network for NER tagging optimized by gradient descent back propagation

# You may try CRF++ tool for NER tagging by CRF model

# You may try other traditional chinese word embedding (ex. fasttext, bert, ...) for input features

# You may try add other features for NER model, ex. POS-tag, word_length, word_position, ...

# You should upload the prediction output on development data or test data provided later to the # competition system.

# Note don't upload prediction output on the splitted testing dataset like this # baseline example.