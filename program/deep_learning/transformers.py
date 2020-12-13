# %%
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))

train_file_path = "../../dataset/train.txt"
test_file_path = "../../dataset/development.txt"

from program.utils.crf_preprocessing import loadInputFile, loadTestFile, CRFFormatData

trainingset, position, mentions = loadInputFile(train_file_path)
testset = loadTestFile(test_file_path)

train_data_path = "../../dataset/crf_data/train.data"
CRFFormatData(trainingset, train_data_path, position)

test_data_path = "../../dataset/crf_data/test.data"
CRFFormatData(testset, test_data_path)

# TODO: check input format

# TODO: check required output format

# %%
