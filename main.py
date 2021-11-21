import sys
from typing import List
from emotionmeter_anew2017 import EmotionMeterANEW2017
from utils import get_logger, set_seed

logger = get_logger('main', True)
set_seed(2021)


def launch(data_path, lexicon_path, output_path):
    meter_test = EmotionMeterANEW2017(data_path=data_path,
                                      lexicon_path=lexicon_path)
    meter_test.load_data()
    meter_test.load_lexicon()
    # print(meter_test._calculate_score(sample_text))
    logger.debug('Executing calculate_score()')
    meter_test.calculate_score()
    logger.debug('Executing calculate_num_token()')
    meter_test.calculate_num_token(meter_test.result_df)
    logger.debug('Executing detect_lang()')
    meter_test.detect_lang(meter_test.result_df)
    logger.debug('Executing detect_hashtag()')
    meter_test.detect_hashtag(meter_test.result_df)
    logger.debug('Executing detect_num_hashtag()')
    meter_test.detect_num_hashtag(meter_test.result_df)
    meter_test.save_score(file_name_or_path=output_path)


def launch_text(text, lexicon_path):
    meter_test = EmotionMeterANEW2017(lexicon_path=lexicon_path)
    meter_test.load_lexicon()
    print(meter_test.calculate_score_text(text))


def main(args: List[str]):
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    default_data_path = "data/tweets/ExtractedTweets.csv"
    default_lexicon_path = "lexicon/anew2017/ANEW2017All.txt"
    default_output_path = "tweets_valence_arousal.csv"
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
