# Baseline Implementation

* [Baseline Implementation](#baseline-implementation)
  * [CRF (Conditional Random Field)](#crf-conditional-random-field)
  * [Data Preprocessing](#data-preprocessing)
    * [data_list](#data_list)
    * [Embed data_list](#embed-data_list)
    * [X, y](#x-y)
  * [Train](#train)
    * [Results](#results)
    * [Upload](#upload)

## CRF (Conditional Random Field)

CRF 模型需要把資料 (**.txt**) 先轉成 CRF 能接受的格式 (**.data**):

| Field | Explain                                        |
| ----- | ---------------------------------------------- |
| O     | 代表無關 NER 的單字                            |
| B-OOO | 代表某個類別 OOO 的起點                        |
| I-OOO | 代表某個類別 OOO 的文字 (可能是文字途中或結尾) |

程式中的 `CRFFormatData()` 會得到一個 `.data` 檔案，裡面包含所有所有 content 每個字對應的 `IOB type`

```
醫 O
師 O
： O
你 O
有 O
...
...
6 B-med_exam
8 I-med_exam
...
...
```

## Data Preprocessing

將上面的 data 檔案透過 `Dataset()` 分成訓練和測試集 (train / test):

| Object                     | Format                                     | Desc                                                          |
| -------------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| data_list                  | (content, sentence, tuple(word, CRF-type)) | 將每個 content 的 data 抓出來，每個 tuple 是每個字的 CRF 資料 |
| train_data_list            | (content, sentence, tuple(word, CRF-type)) | 從 data_list 拆出的 training set                              |
| test_data_list             | (content, sentence, tuple(word, CRF-type)) | 從 data_list 拆出的 testing set                               |
| train_data_article_id_list | (content_id)                               | training set 裡面的對應 content 編號                          |
| test_data_article_id_list  | (content_id)                               | testing set 裡面的對應 content 編號                           |

### data_list

``` python
# shape=(content_len, sentence_len, 1)
[[
  ('他', 'O'),
  ('說', 'O'),
  ('那', 'O'),
  ...
],
...
[
  ('兩', 'B-money'),
  ('百', 'I-money'),
  ('多', 'I-money'),
]]
```

### Embed data_list

將每個文字 (`tuple[0]`) 利用 `Word2Vector` 轉為 512 維度的 embedding 表示:

``` python
train_embed_list = Word2Vector(traindata_list, word_vecs)
[[
  [-5.217900e-02 -1.332038e+00  8.813320e-01 -7.293170e-01 -9.045000e-01 ...] # 他
  [-2.194207e+00 -1.011986e+00  8.231450e-01 -1.624110e-01 -5.918380e-01 ...] # 說
  [-3.255907e+00 -2.022559e+00  8.332450e-02 -2.634220e-01 -6.559380e-01 ...] # 那
],
...
[
  ...
]]
```

### X, y

* `Feature()` 會將每個 512 維度的文字 embedding 轉為字典表示
* `Preprocess()` 將每個 data_list 的 label 提取出來變成 list

``` py
x_train = Feature(train_embed_list)

# an example of a word token 
[
  -5.217900e-02 -1.332038e+00  8.813320e-01 -7.293170e-01 -9.045000e-01 ... 1.895004e+00
]
=>
[
  'dim_1': -0.052179, 'dim_2': -1.332038, 'dim_3': 0.881332, 'dim_4': -0.729317, 'dim_5': -0.9045, ..., 'dim_512': 1.895004
]

y_train = Preprocess(traindata_list)
['O', 'O', 'O', 'B-location', 'I-location', 'O', 'O', 'O', 'O', 'B-location', 'I-location', ...]
```

以下是使用 sample 資料集得到的範例資料格式大小:

| Properties | Shape           | Explain                            |
| ---------- | --------------- | ---------------------------------- |
| x_train    | (17, 1759, 512) | (content, max_len_sentence, embed) |
| y_train    | (17, 1759, 1)   | (content, max_len_sentence, label) |
| x_test     | (9, 2829, 512)  | (content, max_len_sentence, embed) |
| y_test     | (9, 2829, 1)    | (content, max_len_sentence, label) |

## Train

``` python
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
    
crf.fit(x_train, y_train)
y_pred = crf.predict(x_test)
# return predict class
# e.g., B-location

y_pred_mar = crf.predict_marginals(x_test)
# return predict distribution
# e.g., 
# {
#  'O': 0.7364770082815129, 
#  'B-location': 2.253738319713659e-07, 
#  'I-location': 0.2629652437709382, 
#  'B-time': 1.4817077986266331e-06, 
#  'I-time': 8.182176401083988e-05, 
#  'B-med_exam': 4.638799236990415e-06, 
#  'I-med_exam': 0.0003360698810705325, 
#  'B-name': 4.943808713340432e-07, 
#  'I-name': 1.1926458997191943e-06,
#  'B-money': 1.6409408539571999e-06, 
#  'I-money': 0.00013018245396879223
# }

labels = list(crf.classes_)
labels.remove('O')
f1score = metrics.flat_f1_score(
        y_test, y_pred, average='weighted', labels=labels)
```

### Results

| Properties | Format        | Explain                                         |
| ---------- | ------------- | ----------------------------------------------- |
| y_pred     | (9, 2829, 1)  | (content, max_len_sentence, label)              |
| y_pred_mar | (9, 2829, 11) | (content, max_len_sentence, label_distribution) |

### Upload

最終我們要輸出的格式要依照下列表格為準，我們要自己從 `y_pred` 和 `data_list` 去抓出以下格式。

| article_id | start_position | end_position | entity_text | entity_type |
| ---------- | -------------- | ------------ | ----------- | ----------- |
| 8          | 52             | 54           | 前天        | time        |
| 8          | 68             | 70           | 昨天        | time        |
| 8          | 189            | 193          | 二十分鐘    | time        |