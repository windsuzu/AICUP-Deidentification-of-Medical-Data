from program.data_process.default_data_generator import DefaultDataGenerator
from program.data_process.split_data_generator import SplitDataGenerator
from absl import app, flags
from pathlib import Path


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "RAW_TRAIN_DATA_PATH", "dataset/raw_data/train.data", "raw train data output path"
)
flags.DEFINE_string(
    "RAW_TEST_DATA_PATH", "dataset/raw_data/test.data", "raw test data output path"
)

flags.DEFINE_string(
    "TRAIN_DATA_PATH", "dataset/ner_data/train.data", "train data output path"
)
flags.DEFINE_string(
    "TEST_DATA_PATH", "dataset/ner_data/test.data", "test data output path"
)

flags.DEFINE_string("OUTPUT_TYPE", "default", "output type: 1. default, 2. split")


def main(_):
    if not Path(FLAGS.TRAIN_DATA_PATH).parent.exists():
        Path(FLAGS.TRAIN_DATA_PATH).parent.mkdir()

    output_type = ["default", "split"]

    if FLAGS.OUTPUT_TYPE == output_type[1]:
        dataGenerator = SplitDataGenerator()
    else:
        dataGenerator = DefaultDataGenerator()

    dataGenerator.outputTrainData(FLAGS.RAW_TRAIN_DATA_PATH, FLAGS.TRAIN_DATA_PATH)
    dataGenerator.outputTestData(FLAGS.RAW_TEST_DATA_PATH, FLAGS.TEST_DATA_PATH)


if __name__ == "__main__":
    app.run(main)