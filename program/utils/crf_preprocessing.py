import os


def loadInputFile(path):
    # store trainingset [content,content,...]
    trainingset = list()
    # store position[article_id] = [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    position = dict()    
    # store mention[entity_text] = entity_type                       
    mentions = dict()

    with open(path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]

    for data in datas:
        data = data.split('\n')
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]
        separate_position = []

        for annot in annotations[1:]:
            # annot = article_id, start_pos, end_pos, entity_text, entity_type
            annot = annot.split('\t')
            separate_position.extend(annot)
            mentions[annot[3]] = annot[4]
        position[int(separate_position[0])] = separate_position
        
    return trainingset, position, mentions

def loadTestFile(path):
    # store testset [content,content,...]
    testset = list()
    
    with open(path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
        datas = file_text.split('\n\n--------------------\n\n')[:-1]
        for data in datas:
            data = data.split('\n')
            testset.append(data[1])

    return testset


class DiscoveryEmptyCharacter(BaseException):
    pass

def CheckUselessCharacter(article_id, content):
    content_split = set(content)
    if '' or ' ' in content_split:
        raise DiscoveryEmptyCharacter("'' or ' ' in {}th content".format(article_id))

def GenerateLabel(token, token_idx, entity_type):
    # BIO states
    if token_idx == 0:
        label = 'B-' + entity_type
    else:
        label = 'I-' + entity_type
            
    label_str = token + ' ' + label + '\n'
    return label_str

def FillUpZero(token_list, outputfile):
    for token_idx in range(len(token_list)):
        output_str = token_list[token_idx] + ' ' + 'O' + '\n'
        outputfile.write(output_str)

def GenerateFormatData(content, path, position=0):
    
    if (os.path.isfile(path)):
        print("Have been generated")
        return

    outputfile = open(path, 'w', encoding= 'utf-8')
    state = "train" if position else "test"
    
    if state == "test":
        for article_id in range(len(content)):
            CheckUselessCharacter(article_id, content[article_id])

            content_split = "\n".join([word for word in content[article_id]])
            outputfile.write(content_split)
            outputfile.write("\n\n")

            if article_id % 10 == 0:
                print('Total complete articles:', article_id)
    else:
        for article_id in range(len(content)):
            CheckUselessCharacter(article_id, content[article_id])
            
            start_tmp = 0
            separate_position = position[article_id]
            
            for position_idx in range(0, len(separate_position), 5):
                label_start_pos = int(separate_position[position_idx + 1])
                label_end_pos = int(separate_position[position_idx + 2])
                label_entity_type = separate_position[position_idx + 4]
                label_token = list(content[article_id][label_start_pos:label_end_pos])

                begin_idx = 0 if position_idx == 0 else start_tmp
                if label_start_pos != 0:
                    front_token = list(content[article_id][begin_idx:label_start_pos])
                    FillUpZero(front_token, outputfile)
                
                for token_idx in range(len(label_token)):
                    output_str = GenerateLabel(label_token[token_idx], token_idx, label_entity_type)
                    outputfile.write(output_str)
                start_tmp = label_end_pos

            remaining_token = list(content[article_id][start_tmp:])
            FillUpZero(remaining_token, outputfile)

            output_str = '\n'
            outputfile.write(output_str)

            if article_id % 10 == 0:
                print('Total complete articles:', article_id)
    # close output file

    outputfile.close()