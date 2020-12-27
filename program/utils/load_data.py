import os
from program.abstracts.abstract_data_generator import DataGenerator


def loadInputFile(path):
    # store trainingset [content,content,...]
    trainingset = list()
    # store position[article_id] = [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    position = dict()
    # store mention[entity_text] = entity_type
    mentions = dict()

    with open(path, "r", encoding="utf8") as f:
        file_text = f.read().encode("utf-8").decode("utf-8-sig")
    datas = file_text.split("\n\n--------------------\n\n")[:-1]

    for data in datas:
        data = data.split("\n")
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]
        separate_position = []

        for annot in annotations[1:]:
            # annot = article_id, start_pos, end_pos, entity_text, entity_type
            annot = annot.split("\t")
            separate_position.extend(annot)
            mentions[annot[3]] = annot[4]
        position[int(separate_position[0])] = separate_position

    return trainingset, position, mentions


def loadTestFile(path):
    # store testset [content,content,...]
    testset = list()

    with open(path, "r", encoding="utf8") as f:
        file_text = f.read().encode("utf-8").decode("utf-8-sig")
        datas = file_text.split("\n\n--------------------\n\n")[:-1]
        for data in datas:
            data = data.split("\n")
            testset.append(data[1])

    return testset


def blankSpaceExist(token):
    if len(token.replace(" ", "")) == 0:
        return True
    else:
        return False


def generateLabelString(token, token_idx, entity_type):
    # BIO states
    if token_idx == 0:
        label = "B-" + entity_type
    else:
        label = "I-" + entity_type

    label_str = token + " " + label + "\n"
    return label_str


def fillUpZero(token_list, outputfile):
    for token_idx in range(len(token_list)):
        output_str = token_list[token_idx] + " " + "O" + "\n"
        outputfile.write(output_str)


def generateCRFFormatData(content, path, position=0, delete_blank=False):

    if os.path.isfile(path):
        print("Have been generated")
        return

    outputfile = open(path, "w", encoding="utf-8")
    state = "train" if position else "test"

    if state == "test":
        for article_id in range(len(content)):
            if delete_blank:
                clear_content = content[article_id].replace(" ", "")
                content_split = "\n".join([word for word in clear_content])
            else:
                content_split = "\n".join([word for word in content[article_id]])
            outputfile.write(content_split)
            outputfile.write("\n\n")

            if article_id % 10 == 0:
                print("Total complete articles:", article_id)
    else:
        for article_id in range(len(content)):
            start_tmp = 0
            separate_position = position[article_id]

            for position_idx in range(0, len(separate_position), 5):
                label_start_pos = int(separate_position[position_idx + 1])
                label_end_pos = int(separate_position[position_idx + 2])
                label_entity_type = separate_position[position_idx + 4]
                label_token = list(content[article_id][label_start_pos:label_end_pos])

                begin_idx = 0 if position_idx == 0 else start_tmp
                if label_start_pos != 0:
                    front_token = list(
                        content[article_id][begin_idx:label_start_pos].replace(" ", "")
                    )
                    fillUpZero(front_token, outputfile)

                for token_idx in range(len(label_token)):
                    if blankSpaceExist(label_token[token_idx]):
                        continue

                    output_str = generateLabelString(
                        label_token[token_idx], token_idx, label_entity_type
                    )
                    outputfile.write(output_str)
                start_tmp = label_end_pos

            remaining_token = list(content[article_id][start_tmp:].replace(" ", ""))
            fillUpZero(remaining_token, outputfile)

            output_str = "\n"
            outputfile.write(output_str)

            if article_id % 10 == 0:
                print("Total complete articles:", article_id)

    # close output file
    outputfile.close()


class DefaultDataGenerator(DataGenerator):
    def outputTrainData(self, raw_train, output_train):
        trainingset, position, mentions = loadInputFile(raw_train)
        generateCRFFormatData(trainingset, output_train, position)
        print("Default train data generated.")

    def outputTestData(self, raw_test, output_test):
        testingset = loadTestFile(raw_test)
        generateCRFFormatData(testingset, output_test)
        print("Default test data generated.")
