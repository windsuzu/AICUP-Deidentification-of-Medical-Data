from pathlib import Path
from program.data_process.data_preprocessor import GeneralDataPreprocessor
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "TRAIN_DATA_PATH", "dataset/ner_data/train_grained.data", "train data output path"
)
flags.DEFINE_string(
    "TEST_DATA_PATH", "dataset/ner_data/test_grained.data", "test data output path"
)
flags.DEFINE_string(
    "RAW_TEST_DATA_PATH", "dataset/raw_data/test.txt", "raw test data output path"
)
flags.DEFINE_string("MODEL_DATA_PATH", "model/baseline/data/", "model data output path")


def main(_):
    if not Path(FLAGS.MODEL_DATA_PATH).exists():
        Path(FLAGS.MODEL_DATA_PATH).mkdir(parents=True)
    
    dataPreprocessor = GeneralDataPreprocessor(
        FLAGS.TRAIN_DATA_PATH, FLAGS.TEST_DATA_PATH
    )

    dataPreprocessor.outputTrainArrays(
        FLAGS.MODEL_DATA_PATH + "train_X.pkl", FLAGS.MODEL_DATA_PATH + "train_y.pkl"
    )

    dataPreprocessor.outputTestArray(
        FLAGS.MODEL_DATA_PATH + "test_X.pkl",
        FLAGS.MODEL_DATA_PATH + "test_mapping.pkl",
        FLAGS.RAW_TEST_DATA_PATH
    )


if __name__ == "__main__":
    app.run(main)