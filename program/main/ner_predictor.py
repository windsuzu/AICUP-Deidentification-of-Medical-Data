from pathlib import Path
from program.predictor.predictor_bert_crf import BertCrfPredictor
from program.predictor.predictor_bert_bilstm_crf import BertBilstmCrfPredictor
from program.predictor.predictor_bilstm_crf import BilstmCrfPredictor
import tensorflow as tf
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("MODEL", "CRF", "model")
flags.DEFINE_string("MODEL_DATA_PATH", "model/CRF/data/", "model train data path")
flags.DEFINE_string("MODEL_OUTPUT_PATH", "model/CRF/output/", "model output path")
flags.DEFINE_string("MODEL_CHECKPOINT_PATH", "model/CRF/checkpoint/", "checkpoint path")
flags.DEFINE_integer("EMBEDDING_SIZE", 512, "embedding size")
flags.DEFINE_integer("HIIDEN_NUMS", 512, "hidden nums")
flags.DEFINE_float("LEARNING_RATE", 1e-3, "learning rate")

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def buildBiLstmCrfPredictor():
    if FLAGS.MODEL == "BILSTM_CRF":
        return BilstmCrfPredictor(
            FLAGS.MODEL_DATA_PATH,
            FLAGS.MODEL_CHECKPOINT_PATH,
            FLAGS.MODEL_OUTPUT_PATH,
            FLAGS.EMBEDDING_SIZE,
            FLAGS.HIIDEN_NUMS,
            FLAGS.LEARNING_RATE,
        )


def buildBertBiLstmCrfPredictor():
    if FLAGS.MODEL == "BERT_BILSTM_CRF":
        return BertBilstmCrfPredictor(
            FLAGS.MODEL_DATA_PATH,
            FLAGS.MODEL_CHECKPOINT_PATH,
            FLAGS.MODEL_OUTPUT_PATH,
            FLAGS.EMBEDDING_SIZE,
            FLAGS.HIIDEN_NUMS,
            FLAGS.LEARNING_RATE,
        )


def buildBertCrfPredictor():
    if FLAGS.MODEL == "BERT_CRF":
        return BertCrfPredictor(
            FLAGS.MODEL_DATA_PATH,
            FLAGS.MODEL_CHECKPOINT_PATH,
            FLAGS.MODEL_OUTPUT_PATH,
            FLAGS.EMBEDDING_SIZE,
            FLAGS.HIIDEN_NUMS,
            FLAGS.LEARNING_RATE,
        )


def main(_):
    if not Path(FLAGS.MODEL_OUTPUT_PATH).exists():
        Path(FLAGS.MODEL_OUTPUT_PATH).mkdir(parents=True)

    predictor_list = {
        "CRF": None,
        "BILSTM_CRF": buildBiLstmCrfPredictor(),
        "BERT_CRF": buildBertCrfPredictor(),
        "BERT_BILSTM_CRF": buildBertBiLstmCrfPredictor(),
    }

    predictor = predictor_list[FLAGS.MODEL]
    predictor.run()


if __name__ == "__main__":
    app.run(main)