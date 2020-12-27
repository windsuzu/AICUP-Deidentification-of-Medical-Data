# from absl import app, flags

# FLAGS = flags.FLAGS

# flags.DEFINE_integer("EPOCH", 32, "define epoch number")
# flags.DEFINE_float("LR", 1e-5, "define learning rate")

# def main(_):
#     print("Run with epochs:", FLAGS.EPOCH, " lr:", FLAGS.LR)

# if __name__ == "__main__":
#     app.run(main)

import numpy as np

arr = []
arr.append([1,2,3])
arr.append([1,2,3])
arr.append([[1, 2, 3]])
print(arr)