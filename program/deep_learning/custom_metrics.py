import tensorflow as tf

from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(Metrics, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data)[0], -1)
        val_targ = np.concatenate([y for x, y in self.validation_data], axis=0)

        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average="macro")
        _val_recall = recall_score(val_targ, val_predict, average="macro")
        _val_precision = precision_score(val_targ, val_predict, average="macro")

        logs["val_f1"] = _val_f1
        logs["val_recall"] = _val_recall
        logs["val_precision"] = _val_precision
        print(
            " — val_f1: %f — val_precision: %f — val_recall: %f"
            % (_val_f1, _val_precision, _val_recall)
        )
        return
