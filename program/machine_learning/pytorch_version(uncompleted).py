# %%
import os
import pickle
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from utils.load_data import LoadInputFile, LoadTestFile

# Set fixed seed
torch.manual_seed(1)

train_file_path = "data/train_2_half.txt"
trainingset, position, mentions = LoadInputFile(train_file_path)

# %%
def generateWordDict(trainingset):
    word_set = set()
    for content_idx in range(len(trainingset)):
        for word in trainingset[content_idx]:
            if ('\u4e00' <= word <= '\u9fa5') and (word not in word_set):
                word_set.add(word)

    word_dict = {}
    word_dict["</p>"] = 0 # padding
    word_dict["</u>"] = 1 # untoken
    word_dict["</s>"] = 2 # space
    word_dict["</e>"] = 3 # english
    word_dict["</n>"] = 4 # number
    word_dict["</y>"] = 5 # symbol

    for v, k in enumerate(word_set, 6):
        word_dict[k] = v
    
    return word_dict

# separate statement[[[begin_idx, paragraph], [begin_idx, paragraph]],[]...]
import string
additional = {"…", "、"}
punc = set(string.punctuation) | additional

def postProcess(paragraph_list):
    end_idx = 0
    for token_idx in range(len(paragraph_list)-1, 0, -1):
        if paragraph_list[token_idx] in punc:
            end_idx = token_idx + 1
            break
    if len("".join(paragraph_list[1:end_idx])) == 0:
        return paragraph_list
    return [paragraph_list[0], "".join(paragraph_list[1:end_idx])]

def splitContent(trainingset):
    total_splited_content = []
    for content_idx in range(len(trainingset)):

        colon_count = 0
        past_colon_count = 0
        paragraph = None
        separate_splited_content = []

        for word_idx in range(len(trainingset[content_idx])):
            past_colon_count = colon_count
            word = trainingset[content_idx][word_idx]

            if word == ":":
                colon_count += 1
                paragraph_start_idx = word_idx + 1
                if paragraph is not None:
                    paragraph = postProcess(paragraph)
                    separate_splited_content.append(paragraph)
                paragraph = [paragraph_start_idx]


            if colon_count == past_colon_count and paragraph is not None:
                paragraph.append(word)
        else:
            paragraph = postProcess(paragraph)
            separate_splited_content.append(paragraph)

        total_splited_content.append(separate_splited_content)
    return total_splited_content

# %%
# change word to number
def getInputData(total_splited_content,word_dict):
    total_number_paragraph = []
    for content_idx in range(len(total_splited_content)):
        separate_content = total_splited_content[content_idx]

        for para_idx in range(len(separate_content)):
            separate_number_paragraph = []

            for word in separate_content[para_idx][1]:
                try:
                    separate_number_paragraph.append(word_dict[word])
                except:
                    if word == " ":
                        annot = 2
                    elif word.isalpha():
                        annot = 3
                    elif word.isdigit():
                        annot = 4
                    elif word in punc:
                        annot = 5
                    else:
                        annot = 1
                    separate_number_paragraph.append(annot)

            separate_number_paragraph = torch.tensor(separate_number_paragraph, dtype=torch.long)
            total_number_paragraph.append(separate_number_paragraph)
            
    return total_number_paragraph

tags = {'profession':0, 'family':1, 'clinical_event':2, 'ID':3, 'contact':4, 'med_exam':5, 'location':6,
        'organization':7, 'name':8, 'money':9, 'time':10, 'others':11, 'education':12, "<START>": 13, "<STOP>": 14}

def getLabelData(splited_content, position):
    total_label = []

    for content_idx in range(len(splited_content)):
        label_idx = 0
        label_list = position[content_idx]

        for paragraph in splited_content[content_idx]:        
            para_label_list = []
            para_base_idx = paragraph[0]

            for word_idx in range(len(paragraph[1])):
                current_idx = para_base_idx + word_idx
                if label_idx == len(label_list):
                    break
                if current_idx < int(label_list[label_idx+1]) and current_idx < int(label_list[label_idx+2]):
                    para_label_list.append(0)
    #             elif (para_base_idx + word_idx) == int(label_list[label_idx+1]):
    #                 para_label_list.append(tag_to_ix[label_list[label_idx+4]])
                elif current_idx >= int(label_list[label_idx+1]) and current_idx < int(label_list[label_idx+2]):
                    para_label_list.append(tags[label_list[label_idx+4]])

                    if current_idx+1 == int(label_list[label_idx+2]):
                        label_idx += 5

            time = len(paragraph[1]) - len(para_label_list)
            for _ in range(time):
                para_label_list.append(0)

            para_label_list = torch.tensor(para_label_list, dtype=torch.long)
            total_label.append(para_label_list)
    return total_label

# %%
word_dict = generateWordDict(trainingset)
splited_content = splitContent(trainingset)
x_train = getInputData(splited_content, word_dict)
y_train = getLabelData(splited_content, position)

# %%
# embedding matrix
embedding_vector_path = 'data/cna.cbow.cwe_embedding.matrix_512d'

if (os.path.isfile(embedding_vector_path)):
    print("Embedding matrix has been generated")
    with open(embedding_vector_path,'rb') as f:
        embedding_matrix = pickle.load(f)
else:
    # Parse the unzipped file (a .txt file) to build an index that maps 
    # words (as strings) to their vector representation (as number vectors)
    pretrained_word2vec_path = 'data/cna.cbow.cwe_p.tar_g.512d.0.txt'
    embeddings_index = {}
    f = open(pretrained_word2vec_path, encoding='utf8')
    for line in f:
        values = line.strip().split()
        if len(values) == 2:
            continue
        
        word = ''.join(values[:-512])
        coefs = np.asarray(values[-512:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    # Preparing the GloVe word-embeddings matrix
    max_words = len(word_dict)
    embedding_matrix = np.zeros((max_words, 512))
    for word, i in word_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_matrix = torch.FloatTensor(embedding_matrix)
    with open(embedding_vector_path,'wb') as f:
        pickle.dump(embedding_matrix, f)

weight = torch.FloatTensor(embedding_matrix)

# %%
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# word-->number-->tensor
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, self.embedding_dim)
        self.word_embeds.weight = torch.nn.Parameter(weight)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).cuda(),
                torch.randn(2, 1, self.hidden_dim // 2).cuda())

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var.cuda() + trans_score.cuda() + emit_score.cuda()
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1).cuda()
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).cuda(), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def train(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        model.cuda()
        lstm_feats = self._get_lstm_features(sentence)

        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

# %%
EMBEDDING_DIM = 512
HIDDEN_DIM = 32
START_TAG = "<START>"
STOP_TAG = "<STOP>"

model = BiLSTM_CRF(len(word_dict), tags, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

model.to(device)
for epoch in range(20):
    print(epoch+1)
    # again, normally you would NOT do 300 epochs, it is toy data
    for x_batch, y_batch in train_loader:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        if idx % 10000 == 0:
            print("Total complete articles:", idx)
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = x_train[idx].to(device)
        targets = y_train[idx].to(device)
        # Step 3. Run our forward pass.
        neg_log_likelihood_loss = model.train(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        neg_log_likelihood_loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), "model.100.epoch")

# Check predictions after training
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     print(model(precheck_sent))