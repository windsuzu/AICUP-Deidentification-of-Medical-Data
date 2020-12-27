
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("MODEL", "model/baseline/data", "model data output path")


def main(_):
    print(FLAGS.MODEL, "@@")


if __name__ == "__main__":
    app.run(main)