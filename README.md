- [ ] Clear Baseline Program, can explain the details about all the steps, upload result
- [ ] Add other machine learning methods (opt)
- [ ] Add embedding mechanism
  - [ ] Word2Vec
  - [ ] GloVe
  - [ ] FastText
- [ ] Add deep learning methods
  - [ ] basic RNN
  - [ ] BERT
- [ ] Visualization
- [ ] Design Pattern

# AICUP - Deidentification of medical data

## Dataset

NER 是一個分類問題：17 個類別 + other

| Field                                    | Description                                                                                              |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 名字（name）                             | 所有的姓名、綽號、社群/通訊軟體使用者名稱、個人於團體中的代號等。                                        |
| 地點（location）                         | 所有地址、商店名、建築物名稱、景點等。                                                                   |
| 時間（time）                             | 所有日期、時間、年齡等，例如：出生年月日、看診時間。                                                     |
| 聯絡方式（contact）                      | 所有電話號碼、傳真號碼、信箱、IP 位址、網址、網站名稱(例如成大醫院掛號系統)等。                          |
| 編號（id）                               | 所有跟個人有關的編號，例如：身分證號碼、證件號碼、卡號、病歷號等。                                       |
| 職業（profession）                       | 所有任職公司名稱、任職單位等。                                                                           |
| 個人生物標誌（biomarker）                | 所有個人的特殊身體或生理特徵，例如：胎記/疤痕/刺青部位或形狀、植入物(例如人工髖關節、心導管)等。         |
| 家庭成員（family）                       | 所有個人的家庭成員關係，例如：爸爸、姊姊、兒子等。                                                       |
| 有名的臨床事件（clinical_event）         | 所有廣為人知的臨床事件，例如：八仙塵爆、COVID-19。                                                       |
| 特殊專業或技能（special_skills）         | 所有個人獨特的專業或技能，例如：手繪電影看板。                                                           |
| 獨家或聞名的治療方法（unique_treatment） | 所有特別或廣為人知的治療方法，例如：台大醫院葉克膜、長庚醫院甲狀腺射頻消融手術。                         |
| 帳號（account）                          | 所有帳號，例如：社群/通訊軟體帳號或 ID、郵局銀行帳號。                                                   |
| 所屬團體（organization）                 | 所有個人參與的組織、團體、社團等等的名稱，例如：歡樂無法黨、成大教職男籃隊。                             |
| 就學經歷或學歷（education）              | 所有個人的就學經歷或學歷，如系所、程度，例如：讀成大資工、成大資工所碩士畢業。                           |
| 金額（money）                            | 所有金額，例如：看診金額、個人負擔金額、自費金額。                                                       |
| 所屬品的特殊標誌（belonging_mark）       | 所有個人的所屬品特殊標誌，例如：汽車貼膜圖案、產品序列號、手機殼圖案、顏色。                             |
| 報告數值（med_exam）                     | 醫療檢查報告、影像報告的數值，例如：肝功能 67、紅血球值 5.8、超音波影像的脾藏 10.67 公分、體溫 36.7 度。 |
| 其他（others）                           | 其他跟個人隱私有關，可以關聯到當事人的內容。                                                             |

Content:

```
醫師：你有做超音波嘛，那我們來看報告，有些部分有紅字耶。民眾：紅字是甚麼意思？醫師：就是肝功能有比較高，肝功能68，就是這個ALP是68，這樣比較高，正常應是50以下，另外就是你之前說你有B肝，但是你B肝已經好了耶。民眾：它會自動修復阿。醫師：你有抗體了阿，所以你B肝已經沒帶原了耶。民眾：我以前被關的時候，就有在固定驗血，那時候說有B肝。......
```

| article_id | start_position | end_position | entity_text | entity_type |
| ---------- | -------------- | ------------ | ----------- | ----------- |
| 0          | 55             | 57           | 68          | med_exam    |
| 0          | 1264           | 1271         | 10.78公分   | med_exam    |
| 0          | 1358           | 1361         | 三多路      | location    |
| 0          | 1374           | 1378         | 長庚醫院    | location    |
| 0          | 1863           | 1865         | 十天        | time        |
| 0          | 2072           | 2076         | 打撲克牌    | profession  |

程式中的 `loadInputFile()` 會回傳以下的資料結構:

``` python
trainingset (list):
[Content, Content, ...]

position (no use nested array):
[0, 55, 57, "68", "med_exam",
 0, 1264 , 1271, "10.78公分", "med_exam",
 0, 1358, 1361, "三多路", "location",
 ...
 ...
]

mentions (dict):
{
    "68": "med_exam",
    "10.78公分": "med_exam",
    "三多路": "location",
    ...
    ...
}
```

## Conditional random field (CRF)

CRF Model 需要把資料 (**txt format**) 先轉成 CRF 的 input 格式 (**data format**)，以下是 CRF 格式:

| Field | Explain                                        |
| ----- | ---------------------------------------------- |
| O     | 代表無關 NER 的單字                            |
| B-OOO | 代表某個類別 OOO 的起點                        |
| I-OOO | 代表某個類別 OOO 的文字 (可能是文字途中或結尾) |

Example: 

```
肝 O
功 O
能 O
6 B-med_exam
8 I-med_exam
```

程式中的 `CRFFormatData()` 會得到一個 data 檔案，裡面包含所有所有 content 每個字對應的 `IOB type`

```
醫 O
師 O
： O
你 O
有 O
做 O
超 O
音 O
波 O
...
...
6 B-med_exam
8 I-med_exam
...
...
```

### Data Preprocessing

首先將上面的 data 檔案透過 `Dataset()` 分成訓練集 (training set) 和測試集 (test set) 得到以下幾個物件:

| Object                     | Format            | Desc                                                          |
| -------------------------- | ----------------- | ------------------------------------------------------------- |
| data_list                  | list(list(tuple)) | 將每個 content 的 data 抓出來，每個 tuple 是每個字的 CRF 資料 |
| train_data_list            | list(list(tuple)) | 從 data_list 拆出的 training set                              |
| test_data_list             | list(list(tuple)) | 從 data_list 拆出的 testing set                               |
| train_data_article_id_list | list()            | training set 裡面的對應 content 編號                          |
| test_data_article_id_list  | list()            | testing set 裡面的對應 content 編號                           |

以下是 `data_list` 的資料格式:

``` python
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

再來將 data_list (train, test) 中的文字 `tuple[0]` 利用 `Word2Vector(data_list, word_vecs)` 來轉為 embedding 表示，會變成 `list(list(list))`，最裡面的 list 是 512 維度的文字 embedding。

``` python
trainembed_list = Word2Vector(traindata_list, word_vecs)
[[
  [-5.217900e-02 -1.332038e+00  8.813320e-01 -7.293170e-01 -9.045000e-01 ...] # 他
  [-2.194207e+00 -1.011986e+00  8.231450e-01 -1.624110e-01 -5.918380e-01 ...] # 說
],
...
[
  ...
]]
```

最後就是分裝 train 和 test 的 data-label pair: 

* `Feature()` 會將每個 512 維度的文字 embedding 轉為字典表示
* `Preprocess()` 將每個 data_list 的 label 提取出來變成 list

以下是使用 sample 資料集得到的範例資料格式大小:

| Properties | Shape           | Explain                            |
| ---------- | --------------- | ---------------------------------- |
| x_train    | (17, 1759, 512) | (content, max_len_sentence, embed) |
| y_train    | (17, 1759, 1)   | (content, max_len_sentence, label) |
| x_test     | (9, 2829, 512)  | (content, max_len_sentence, embed) |
| y_test     | (9, 2829, 1)    | (content, max_len_sentence, label) |


``` python
x_train = Feature(trainembed_list)

# Single token representation
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

### Training & Decoding

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

| Properties | Format        | Explain                                         |
| ---------- | ------------- | ----------------------------------------------- |
| y_pred     | (9, 2829, 1)  | (content, max_len_sentence, label)              |
| y_pred_mar | (9, 2829, 11) | (content, max_len_sentence, label_distribution) |

## Upload

最終我們要輸出的格式要依照下列表格為準，我們要自己從 `y_pred` 和 `data_list` 去抓出以下格式。

| article_id | start_position | end_position | entity_text | entity_type |
| ---------- | -------------- | ------------ | ----------- | ----------- |
| 8          | 52             | 54           | 前天        | time        |
| 8          | 68             | 70           | 昨天        | time        |
| 8          | 189            | 193          | 二十分鐘    | time        |