# %%
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path


test_encodings = tokenizer(
    test_texts,
    is_split_into_words=True,
    padding=True,
    truncation=True,
    return_token_type_ids=False,
)

test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings)))




# %%
prediction = model.predict(test_dataset)[0]
# prediction = np.argmax(prediction, -1)
# batch_size = MAX_SENTENCE_LENGTH
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
