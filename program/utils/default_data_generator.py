from program.utils.load_data import generateCRFFormatData, loadInputFile, loadTestFile
from program.abstracts.abstract_data_generator import DataGenerator


class DefaultDataGenerator(DataGenerator):
    def outputTrainData(self, raw_train, output_train):
        trainingset, position, mentions = loadInputFile(raw_train)
        generateCRFFormatData(trainingset, output_train, position)
        print("Default train data generated.")

    def outputTestData(self, raw_test, output_test):
        testingset = loadTestFile(raw_test)
        generateCRFFormatData(testingset, output_test)
        print("Default test data generated.")
