#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import re


# In[12]:


file_path = 'data/train_2.txt'
annotation_entries = ["article_id", "start_pos", "end_pos", "entity_text", "entity_type", "length"]


# In[13]:


def loadInputFile(path):
    trainingset = list()                             # store trainingset [content,content,...]
    position = list()                                # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    mentions = dict()                                # store mentions[mention] = Type
    
    annotation = dict([(name,[]) for name in annotation_entries])
    
    with open(path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]
    
    for data in datas:
        data = data.split('\n')
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]
        
        for annot in annotations[1:]:
            annot = annot.split('\t')                  # annot = article_id, start_pos, end_pos, entity_text, entity_type
            
            length = int(annot[2]) - int(annot[1])
            annot.append(length)
            
            position.extend(annot)
            mentions[annot[3]] = annot[4]
            
            annotation = ListToDict(annot, annotation)

    return trainingset, position, mentions, annotation

def ListToDict(annot_list, annotation):   
    for i, entry in enumerate(annotation_entries):
        annotation[entry].append(annot_list[i])
    
    return annotation

def ExtractStrangeEntityText():
    texts = []
    regex = r"…+"                                      # 神奇的符號：…

    for text in df.entity_text:
        if re.search(regex, text):
            texts.append(text)
    
    return texts


# In[14]:


trainingSet, position, mentions, annotation = loadInputFile(file_path)

text_length = []
for trainingData in trainingSet:
    text_length.append(len(trainingData))


# In[19]:


df = pd.DataFrame(annotation, columns = list(annotation.keys()))
df


# ### 奇怪的資料集
# * 第十六個對話，第三百零六個註記錯(已訂正)
# * 出現神奇的點點

# In[16]:


# df.iloc[306]
print(*ExtractStrangeEntityText(), sep="\n")


# ### 實體類型分佈狀況
# * 總共18個，只出現13個
# * 沒出現：biomarker, special_skills, unique_treatment, account, belonging_mark
# * 標註的很奇怪

# In[17]:


df.entity_type.value_counts()


# ### 視覺化一下
# * 一篇文章出現幾個註解分佈
# * 一個實體的長度的分佈
# * 文章長度的分佈

# In[18]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[15, 4])

ax1_data = df["article_id"].value_counts()
ax1.hist(ax1_data)
ax1.set_title('annotation amount ({min}, {max})'.format(min=ax1_data.min(), max=ax1_data.max()))

ax2_data = df["length"]
ax2.hist(ax2_data, color="orange")
ax2.set_title('entity length ({min}, {max})'.format(min=ax2_data.min(), max=ax2_data.max()))

ax3_data = text_length
ax3.hist(ax3_data, color="green")
ax3.set_title('text length ({min}, {max})'.format(min=min(ax3_data), max=max(ax3_data)))


# ### 問題
# * 點點點是否要清掉（清掉後帶來的後果）
# * 種類極度不平均
# 
# ### TOTRY
# * 去掉角色、全形半形、統一語言
# * CRF++、Python-crfsuite、Pytorch-crfsuite
