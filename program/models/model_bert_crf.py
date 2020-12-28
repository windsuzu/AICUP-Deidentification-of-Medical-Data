import tensorflow as tf
from tf2crf import CRF, ModelWithCRFLoss


def BertCrfModel(transformer_model, max_sentence_length, label_nums):
    input_ids = tf.keras.Input(
            name="input_ids", shape=(max_sentence_length,), dtype=tf.int32
        )
    attention_mask = tf.keras.Input(
        name="attention_mask", shape=(max_sentence_length,), dtype=tf.int32
    )
    transformer = transformer_model([input_ids, attention_mask])
    hidden_states = transformer[1]

    hidden_states_size = 1
    hidden_states_ind = list(range(-hidden_states_size, 0, 1))

    selected_hidden_states = tf.keras.layers.concatenate(
        tuple([hidden_states[i] for i in hidden_states_ind])
    )
    crf = CRF(dtype="float32")
    output = tf.keras.layers.Dense(label_nums, activation="relu")(
        selected_hidden_states
    )
    output = crf(output)

    model = tf.keras.models.Model(
        inputs=[input_ids, attention_mask], outputs=output
    )
    model = ModelWithCRFLoss(model)
    return model