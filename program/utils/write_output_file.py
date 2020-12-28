def getBlankSpacePosition(total_content):
    total_postion = dict()
    for content_idx in range(len(total_content)):
        separate_position = list()
        separate_content = total_content[content_idx]
        
        for token_idx in range(len(separate_content)):
            if separate_content[token_idx] == " ":
                separate_position.append(token_idx)
                
        if separate_position:
            total_postion[content_idx] = separate_position
            
    return total_postion

def fillUpBlankSpace(y_pred, test_data_list, blank_space_position):
    for content_idx, blank_position_list in list(blank_space_position.items()):
        for blank_position in blank_position_list:
            y_pred[content_idx].insert(blank_position, "0")
            test_data_list[content_idx].insert(blank_position, [" "])
    return y_pred, test_data_list

def checkEndOfLabel(start_pos, separate_content_pred, token_pred_idx):
    if ((start_pos is not None) and
        (separate_content_pred[token_pred_idx][0] == "I") and
        (token_pred_idx + 1 == len(separate_content_pred) or separate_content_pred[token_pred_idx + 1][0] == "O")):
        return True
    else:
        return False
    
def getEntityTxt(separate_content, start_pos, end_pos):
    return "".join([separate_content[token_idx][0] for token_idx in range(start_pos, end_pos)])

def generateOutputFile(y_pred, test_data_list, testingset, output_path, delete_blank):
    title = "article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    output = title
    
    if delete_blank:
        blank_space_position = getBlankSpacePosition(testingset)
        y_pred, test_data_list = fillUpBlankSpace(y_pred, test_data_list, blank_space_position)
    
    for content_pred_idx in range(len(y_pred)):
        pos = 0
        start_pos, end_pos = None, None
        entity_text, entity_type = None, None
        separate_content_pred = y_pred[content_pred_idx]

        for token_pred_idx in range(len(separate_content_pred)):
            if separate_content_pred[token_pred_idx][0] == "B":
                start_pos = pos
                entity_type = separate_content_pred[token_pred_idx][2:]
                
            elif checkEndOfLabel(start_pos, separate_content_pred, token_pred_idx):
                end_pos = pos + 1 
                entity_text = getEntityTxt(test_data_list[content_pred_idx], start_pos, end_pos)
                
                line = (str(content_pred_idx) + "\t" +
                        str(start_pos) + "\t" + str(end_pos) + "\t" +
                        entity_text + "\t" + entity_type)
                output += line + "\n"
            pos += 1


    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
        
        
tag_check = {
    "I": ["B", "I"],
    "E": ["B", "I"],
}

def check_label(front_label, follow_label):
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (
        (follow_label.startswith("I-") or follow_label.startswith("E-"))
        and front_label.endswith(follow_label.split("-")[1])
        and front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]
    ):
        return True
    return False


def format_result(chars, tags):
    """
    將 TEXT 和 TAG 抓出來，回傳 entity 列表。

    Args:
        chars: ['国','家','发','展','计','划','委','员','会','副','主','任','王','春','正']

        tags: ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'E-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'E-PER']

    Returns:
        [{'begin': 0, 'end': 9, 'words': '国家发展计划委员会', 'type': 'ORG'},
         {'begin': 12, 'end': 15, 'words': '王春正', 'type': 'PER'}]
    """

    entities = []
    entity = []
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])
    if entity:
        entities.append(entity)

    entities_result = []
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {
                    "begin": entity[0][0],
                    "end": entity[-1][0] + 1,
                    "words": "".join([char for _, char, _, _ in entity]),
                    "type": entity[0][2].split("-")[1],
                }
            )

    return entities_result