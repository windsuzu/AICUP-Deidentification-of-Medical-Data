from pathlib import Path
from program.trainer.trainer_bert_bilstm_crf import BertBilstmCrfTrainer
from program.trainer.trainer_bert_crf import BertCrfTrainer
from program.trainer.trainer_bilstm_crf import BiLstmCrfTrainer
from absl import app, flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("MODEL", "CRF", "model")
flags.DEFINE_string("TRAIN_DATA_PATH", "dataset/ner_data/train.data", "train data path")
flags.DEFINE_string("MODEL_DATA_PATH", "model/CRF/data/", "model train data path")
flags.DEFINE_string("MODEL_CHECKPOINT_PATH", "model/CRF/checkpoint/", "checkpoint path")
flags.DEFINE_integer("CHECKPOINT_KEEP", 3, "checkpoint max-to-keep")
flags.DEFINE_integer("SENTENCE_MAX_LENGTH", 32, "sentence max length")
flags.DEFINE_integer("BATCH_SIZE", 128, "batch size")
flags.DEFINE_integer("EMBEDDING_SIZE", 512, "embedding size")
flags.DEFINE_integer("HIIDEN_NUMS", 512, "hidden nums")
flags.DEFINE_integer("EPOCHS", 20, "epochs")
flags.DEFINE_float("LEARNING_RATE", 1e-3, "learning rate")
flags.DEFINE_bool("VISUALIZE", True, "visualize or not")

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def buildBiLstmCrfTrainer():
    if FLAGS.MODEL == "BILSTM_CRF":
        return BiLstmCrfTrainer(
            FLAGS.TRAIN_DATA_PATH,
            FLAGS.MODEL_DATA_PATH,
            FLAGS.MODEL_CHECKPOINT_PATH,
            FLAGS.CHECKPOINT_KEEP,
            FLAGS.SENTENCE_MAX_LENGTH,
            FLAGS.BATCH_SIZE,
            FLAGS.EMBEDDING_SIZE,
            FLAGS.HIIDEN_NUMS,
            FLAGS.EPOCHS,
            FLAGS.LEARNING_RATE,
            FLAGS.VISUALIZE,
        )


def buildBertCrfTrainer():
    if FLAGS.MODEL == "BERT_CRF":
        return BertCrfTrainer(
            FLAGS.TRAIN_DATA_PATH,
            FLAGS.MODEL_DATA_PATH,
            FLAGS.MODEL_CHECKPOINT_PATH,
            FLAGS.CHECKPOINT_KEEP,
            FLAGS.SENTENCE_MAX_LENGTH,
            FLAGS.BATCH_SIZE,
            FLAGS.EMBEDDING_SIZE,
            FLAGS.HIIDEN_NUMS,
            FLAGS.EPOCHS,
            FLAGS.LEARNING_RATE,
            FLAGS.VISUALIZE,
        )


def buildBertBilstmCrfTrainer():
    if FLAGS.MODEL == "BERT_BILSTM_CRF":
        return BertBilstmCrfTrainer(
            FLAGS.TRAIN_DATA_PATH,
            FLAGS.MODEL_DATA_PATH,
            FLAGS.MODEL_CHECKPOINT_PATH,
            FLAGS.CHECKPOINT_KEEP,
            FLAGS.SENTENCE_MAX_LENGTH,
            FLAGS.BATCH_SIZE,
            FLAGS.EMBEDDING_SIZE,
            FLAGS.HIIDEN_NUMS,
            FLAGS.EPOCHS,
            FLAGS.LEARNING_RATE,
            FLAGS.VISUALIZE,
        )


def main(_):
    if not Path(FLAGS.MODEL_DATA_PATH).exists():
        Path(FLAGS.MODEL_DATA_PATH).mkdir(parents=True)

    if not Path(FLAGS.MODEL_CHECKPOINT_PATH).exists():
        Path(FLAGS.MODEL_CHECKPOINT_PATH).mkdir(parents=True)

    trainer_list = {
        "CRF": None,
        "BILSTM_CRF": buildBiLstmCrfTrainer(),
        "BERT_CRF": buildBertCrfTrainer(),
        "BERT_BILSTM_CRF": buildBertBilstmCrfTrainer(),
    }

    trainer = trainer_list[FLAGS.MODEL]
    trainer.run()


if __name__ == "__main__":
    app.run(main)