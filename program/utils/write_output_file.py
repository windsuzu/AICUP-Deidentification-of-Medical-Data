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
        
