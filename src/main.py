from multiprocessing import Pool

import numpy as np
import os
import pandas as pd
from pathlib import Path

from emotionmeter import CircumplexEmotionMeter
from utils import get_logger, set_seed

logger = get_logger('main', True)
set_seed(2021)


def async_launch(data_path, lexicon_path, output_path, n_workers=os.cpu_count() - 1):
    _tmp_files_dir = Path('.', 'tmp')
    _full_csv = pd.read_csv(data_path)
    _splitted_csv = np.array_split(_full_csv, n_workers)
    _tmp_results = [str(Path(_tmp_files_dir, f'{i}.csv')) for i in range(n_workers)]

    p = Pool(n_workers)
    for _idx, _df in enumerate(_splitted_csv):
        if _idx == 0:
            use_tqdm = True
        else:
            use_tqdm = False
        p.apply_async(
            launch,
            args=(
                _df,
                lexicon_path,
                _tmp_results[_idx],
                use_tqdm
            )
        )
    p.close()
    p.join()

    result_list = []
    for _p in _tmp_results:
        result_list.append(pd.read_csv(_p))
    result_df = pd.concat(result_list)
    result_df.to_csv(output_path)


def launch(data_path_or_df, lexicon_path, output_path, text_column='Tweet', use_tqdm=False):
    meter_test = CircumplexEmotionMeter(
        data_path_or_df=data_path_or_df,
        lexicon_path=lexicon_path,
        text_column=text_column,
        use_tqdm=use_tqdm
    )
    meter_test.load_data()
    meter_test.load_lexicon()
    meter_test.load_cognition_and_cognition_word_lists()
    meter_test.load_stopwords()
    meter_test.load_tokenizer()
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


def main(args):
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    tweet_data_path = "../data/tweets/ExtractedTweets.csv"
    trump_data_path = "../data/tweets/trump_archive.csv"

    anew_lexicon_path = "../lexicon/ANEW2017/ANEW2017All.txt"
    nrc_lexicon_path = "../lexicon/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt"

    text = argv.get('text', None)

    _data_arg = argv.get('data', 'tweet')

    if _data_arg == 'tweet':
        data_path = tweet_data_path
        text_column = 'Tweet'
    elif _data_arg == 'trump':
        data_path = trump_data_path
        text_column = 'doc'
    else:
        raise NotImplementedError

    _lex_arg = argv.get('lexicon', 'anew')

    if _lex_arg == 'anew':
        lexicon_path = anew_lexicon_path
    elif _lex_arg == 'nrc':
        lexicon_path = nrc_lexicon_path
    else:
        raise NotImplementedError

    output_path = argv.get('output', '../output_' + _data_arg + '.csv')

    if text is not None:
        launch_text(text, lexicon_path)
        exit(0)
    else:
        launch(data_path, lexicon_path, output_path, text_column=text_column, use_tqdm=True)
        exit(0)


if __name__ == "__main__":
    # main(sys.argv)
    default_data_path = "../data/tweets/ExtractedTweets.csv"

    default_lexicon_path = "../lexicon/ANEW2017/ANEW2017All.txt"
    default_output_path = "../output_anew.csv"
    launch(default_data_path, default_lexicon_path, default_output_path, use_tqdm=True)

    # default_lexicon_path = "../lexicon/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt"
    # default_output_path = "../output_nrc.csv"
    # launch(default_data_path, default_lexicon_path, default_output_path, use_tqdm=True)

    # default_data_path = "../data/tweets/trump_archive.csv"

    # # default_lexicon_path = "../lexicon/ANEW2017/ANEW2017All.txt"
    # # default_output_path = "../output_trump_anew.csv"
    # # launch(default_data_path, default_lexicon_path, default_output_path, text_column='doc', use_tqdm=True)
    # #
    # default_lexicon_path = "../lexicon/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt"
    # default_output_path = "../output_trump_nrc.csv"
    # launch(default_data_path, default_lexicon_path, default_output_path, text_column='doc', use_tqdm=True)
