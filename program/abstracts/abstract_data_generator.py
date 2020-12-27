from abc import ABC, abstractmethod


class DataGenerator(ABC):
    @abstractmethod
    def outputTrainData(self, raw_train, output_train):
        ...

    @abstractmethod
    def outputTestData(self, raw_test, output_test):
        ...