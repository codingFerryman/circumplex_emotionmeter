import sys
from typing import List
from emotionmeter_VAD import CircumplexEmotionMeter
from utils import get_logger, set_seed

logger = get_logger('main', True)
set_seed(2021)


def launch(data_path, lexicon_path, output_path, **lexicon_kwargs):
    meter_test = CircumplexEmotionMeter(data_path=data_path,
                                        lexicon_path=lexicon_path,
                                        affection_path="../emotionmeter/word_lists/affect_list.txt",
                                        cognition_path="../emotionmeter/word_lists/cognition_list.txt"
                                        )
    meter_test.load_data()
    meter_test.load_lexicon(**lexicon_kwargs)
    meter_test.load_cognition_and_cognition_word_lists()
    # print(meter_test._calculate_score(sample_text))
    logger.debug('Executing calculate_score()')
    meter_test.calculate_score()
    # logger.debug('Executing calculate_num_token()')
    # meter_test.calculate_num_token(meter_test.result_df)
    # logger.debug('Executing detect_lang()')
    # meter_test.detect_lang(meter_test.result_df)
    # logger.debug('Executing detect_hashtag()')
    # meter_test.detect_hashtag(meter_test.result_df)
    # logger.debug('Executing detect_num_hashtag()')
    # meter_test.detect_num_hashtag(meter_test.result_df)
    meter_test.save_score(file_name_or_path=output_path)


def launch_text(text, lexicon_path):
    meter_test = CircumplexEmotionMeter(lexicon_path=lexicon_path)
    meter_test.load_lexicon()
    print(meter_test.calculate_score_text(text))


def main(args: List[str]):
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    default_data_path = "../data/tweets/ExtractedTweets.csv"

    # default_lexicon_path = "lexicon/ANEW2017/ANEW2017All.txt"
    # default_output_path = "output_anew.csv"
    default_lexicon_path = "../lexicon/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt"
    default_output_path = "../output_nrc.csv"

    text = argv.get('text', None)
    data_path = argv.get('data', default_data_path)
    lexicon_path = argv.get('lexicon', default_lexicon_path)
    output_path = argv.get('output', default_output_path)

    if text is not None:
        launch_text(text, lexicon_path)
        exit(0)
    else:
        launch(data_path, lexicon_path, output_path)


if __name__ == "__main__":
    main(sys.argv)
