from program.abstracts.abstract_data_generator import DataGenerator
import re


def getArticles(data_path, for_predict=False):
    """
    從 train.data 或 test.data (for_predict) 取得完整列表
    """
    articles = []
    article = []
    with open(data_path, encoding="utf8") as f:
        texts = f.readlines()
        for text in texts:
            if text == "\n":
                articles.append(article)
                article = []
                continue

            if for_predict:
                character = text[0]
            else:
                word, symbol = text.split(" ")
                symbol = symbol.strip("\n")
                character = (word, symbol)

            article.append(character)
    return articles


def split_article(article, for_predict=False):
    """
    把含 label 的單篇文章拆成含 label 的多個句子

    Args:
        article: list(tuple)
            [
                ('醫', 'O'),
                ('師', 'O'),
                ('：', 'O'),
                ('一', 'O'),
                ('陣', 'O'),
                ('子', 'O'),
                ('沒', 'O'),
                ('有', 'O'),
                ('。', 'O'),
                ('民', 'O'),
                ('眾', 'O'),
                ('：', 'O'),
                ('對', 'O'),
                ('。', 'O'),
            ]

    Returns:
        grained_articles: list(list(tuple))
            [
                [('醫', 'O'),
                 ('師', 'O'),
                 ('：', 'O'),
                 ('一', 'O'),
                 ('陣', 'O'),
                 ('子', 'O'),
                 ('沒', 'O'),
                 ('有', 'O'),
                 ('。', 'O')],

                [('民', 'O'),
                 ('眾', 'O'),
                 ('：', 'O'),
                 ('對', 'O'),
                 ('。', 'O')]
            ]
    """
    label = []
    if not for_predict:
        article, label = zip(*article)
    article = "".join(article)

    grained_texts = re.split(r"((?<=\W)\w{2,5}：)", article)
    first = grained_texts.pop(0)
    grained_texts = [first] + [
        speaker + sentence
        for speaker, sentence in zip(grained_texts[0::2], grained_texts[1::2])
    ]

    if for_predict:
        return [list(grained) for grained in grained_texts]

    start = 0
    grained_articles = []

    for grained in grained_texts:
        grained_label = label[start : (start + len(grained))]
        article = list(zip(grained, grained_label))
        grained_articles.append(article)
        start += len(grained)

    return grained_articles


def generateCRFGrainedData(data_path, output_data_path, for_predict=False):
    articles = getArticles(data_path, for_predict)

    with open(output_data_path, "w", encoding="UTF8") as f:
        for article in articles:
            for grained_article in split_article(article, for_predict):
                if for_predict:
                    grained = "\n".join([grained for grained in grained_article])
                else:
                    grained = "\n".join(
                        [" ".join(grained) for grained in grained_article]
                    )
                f.write(grained + "\n\n")


class SplitDataGenerator(DataGenerator):
    def outputTrainData(self, raw_train, output_train):
        generateCRFGrainedData(raw_train, output_train)
        print("Split train data generated.")

    def outputTestData(self, raw_test, output_test):
        generateCRFGrainedData(raw_test, output_test, True)
        print("Split test data generated.")
