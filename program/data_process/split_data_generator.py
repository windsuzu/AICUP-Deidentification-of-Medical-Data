from program.abstracts.abstract_data_generator import DataGenerator
from program.utils.split_data import generateCRFGrainedData


class SplitDataGenerator(DataGenerator):
    def outputTrainData(self, raw_train, output_train):
        generateCRFGrainedData(raw_train, output_train)
        print("Split train data generated.")

    def outputTestData(self, raw_test, output_test):
        generateCRFGrainedData(raw_test, output_test, True)
        print("Split test data generated.")