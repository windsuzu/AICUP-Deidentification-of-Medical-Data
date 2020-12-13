import os


def loadInputFile(path):
    trainingset = list()  # store trainingset [content,content,...]
    # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    position = list()
    mentions = dict()  # store mentions[mention] = Type
    with open(path, "r", encoding="utf8") as f:
        file_text = f.read().encode("utf-8").decode("utf-8-sig")
    datas = file_text.split("\n\n--------------------\n\n")[:-1]
    for data in datas:
        data = data.split("\n")
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]
        for annot in annotations[1:]:
            # annot = article_id, start_pos, end_pos, entity_text, entity_type
            annot = annot.split("\t")
            position.extend(annot)
            mentions[annot[3]] = annot[4]

    return trainingset, position, mentions


def loadTestFile(path):
    testset = list()  # store testset [content,content,...]
    with open(path, "r", encoding="utf8") as f:
        file_text = f.read().encode("utf-8").decode("utf-8-sig")
        datas = file_text.split("\n\n--------------------\n\n")[:-1]
        for data in datas:
            data = data.split("\n")
            testset.append(data[1])

    return testset


def CRFFormatData(dataset, path, position=0):
    if os.path.isfile(path):
        os.remove(path)
    outputfile = open(path, "a", encoding="utf-8")

    if not position:
        for article_id in range(len(dataset)):
            testset_split = list(dataset[article_id])
            while "" or " " in testset_split:
                if "" in testset_split:
                    testset_split.remove("")
                else:
                    testset_split.remove(" ")
            content = "\n".join([word for word in dataset[article_id]])
            outputfile.write(content)
            outputfile.write("\n\n")

            if article_id % 10 == 0:
                print("Total complete articles:", article_id)
        outputfile.close()
        return

    # output file lines
    count = 0  # annotation counts in each content
    tagged = list()
    for article_id in range(len(dataset)):
        testset_split = list(dataset[article_id])
        while "" or " " in testset_split:
            if "" in testset_split:
                testset_split.remove("")
            else:
                testset_split.remove(" ")
        start_tmp = 0
        for position_idx in range(0, len(position), 5):
            if int(position[position_idx]) == article_id:
                count += 1
                if count == 1:
                    start_pos = int(position[position_idx + 1])
                    end_pos = int(position[position_idx + 2])
                    entity_type = position[position_idx + 4]
                    if start_pos == 0:
                        token = list(dataset[article_id][start_pos:end_pos])
                        whole_token = dataset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(" ", "")) == 0:
                                continue
                            # BIO states
                            if token_idx == 0:
                                label = "B-" + entity_type
                            else:
                                label = "I-" + entity_type

                            output_str = token[token_idx] + " " + label + "\n"
                            outputfile.write(output_str)

                    else:
                        token = list(dataset[article_id][0:start_pos])
                        whole_token = dataset[article_id][0:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(" ", "")) == 0:
                                continue

                            output_str = token[token_idx] + " " + "O" + "\n"
                            outputfile.write(output_str)

                        token = list(dataset[article_id][start_pos:end_pos])
                        whole_token = dataset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(" ", "")) == 0:
                                continue
                            # BIO states
                            if token[0] == "":
                                if token_idx == 1:
                                    label = "B-" + entity_type
                                else:
                                    label = "I-" + entity_type
                            else:
                                if token_idx == 0:
                                    label = "B-" + entity_type
                                else:
                                    label = "I-" + entity_type

                            output_str = token[token_idx] + " " + label + "\n"
                            outputfile.write(output_str)

                    start_tmp = end_pos
                else:
                    start_pos = int(position[position_idx + 1])
                    end_pos = int(position[position_idx + 2])
                    entity_type = position[position_idx + 4]
                    if start_pos < start_tmp:
                        continue
                    else:
                        token = list(dataset[article_id][start_tmp:start_pos])
                        whole_token = dataset[article_id][start_tmp:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(" ", "")) == 0:
                                continue
                            output_str = token[token_idx] + " " + "O" + "\n"
                            outputfile.write(output_str)

                    token = list(dataset[article_id][start_pos:end_pos])
                    whole_token = dataset[article_id][start_pos:end_pos]
                    for token_idx in range(len(token)):
                        if len(token[token_idx].replace(" ", "")) == 0:
                            continue
                        # BIO states
                        if token[0] == "":
                            if token_idx == 1:
                                label = "B-" + entity_type
                            else:
                                label = "I-" + entity_type
                        else:
                            if token_idx == 0:
                                label = "B-" + entity_type
                            else:
                                label = "I-" + entity_type

                        output_str = token[token_idx] + " " + label + "\n"
                        outputfile.write(output_str)
                    start_tmp = end_pos

        token = list(dataset[article_id][start_tmp:])
        whole_token = dataset[article_id][start_tmp:]
        for token_idx in range(len(token)):
            if len(token[token_idx].replace(" ", "")) == 0:
                continue

            output_str = token[token_idx] + " " + "O" + "\n"
            outputfile.write(output_str)

        count = 0

        output_str = "\n"
        outputfile.write(output_str)
        ID = dataset[article_id]

        if article_id % 10 == 0:
            print("Total complete articles:", article_id)

    # close output file
    outputfile.close()