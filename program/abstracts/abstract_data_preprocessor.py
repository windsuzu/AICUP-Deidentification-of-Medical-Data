from abc import ABC, abstractmethod


class DataPreprocessor(ABC):
    """
    Inputs:
        [train.data]
        [test.data]

    Outputs:
        [train_X.pkl]: 由文字組成的 array
        [train_y.pkl]: 由文字組成的 array
        [test_X.pkl]: 由文字組成的 array
        [test_mapping.pkl]: 由 test 每篇長度組成的 array，用於輸出
        
    Comment:
        Tokenize 的方式交由下一階段的 Trainer 自行調整。
        Preprocessor 僅輸出文字陣列。
    """

    def __init__(self, train_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    @abstractmethod
    def outputTrainArrays(self, train_X_path, train_y_path):
        ...

    @abstractmethod
    def outputTestArray(self, test_X_path, test_mapping_path):
        ...