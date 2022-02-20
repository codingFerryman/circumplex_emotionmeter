from multiprocessing import Pool, freeze_support
import copy
from typing import Union, Optional

import pandas as pd

from emotionmeter import CircumplexEmotionMeter
from utils import get_logger, set_seed

logger = get_logger('main', True)
set_seed(42)


def _launch_init(use_tqdm=False, corpus="en_core_web_lg"):
    """
    Initialize an emotionmeter
    """
    _emotionmeter = CircumplexEmotionMeter(
        data_path_or_df=None,
        lexicon_path=None,
        text_column=None,
        corpus=corpus,
        use_tqdm=use_tqdm
    )
    return _emotionmeter


def launch(emotionmeter: CircumplexEmotionMeter,
           data_path_or_df: Union[str, pd.DataFrame],
           lexicon_path: str,
           output_path: str,
           text_column: str = 'Tweet',
           microsoft_translator_api_key: Optional[str] = None
           ):
    emotionmeter.self_refresh()
    emotionmeter.set_text_column(text_column=text_column)
    emotionmeter.load_tokens(
        data_path_or_df=data_path_or_df,
        ms_translator_key=microsoft_translator_api_key
    )
    emotionmeter.load_lexicon(path=lexicon_path)

    logger.debug('Executing calculate_score()')
    emotionmeter.calculate_score()
    emotionmeter.save_score(file_name_or_path=output_path)
    return emotionmeter


def launch_text(emotionmeter: CircumplexEmotionMeter, text, lexicon_path):
    emotionmeter.load_lexicon(path=lexicon_path)
    print(emotionmeter.calculate_score_text(text))


if __name__ == "__main__":
    microsoft_translator_api_key = None
    # ============
    # Initialize emotionmeter
    # ============
    initialized_meter = _launch_init(use_tqdm=True, corpus='cache/w2v')
    initialized_meter_trump = _launch_init(use_tqdm=True, corpus='cache/w2v_trump')

    # ============
    # Set configurations
    # ============
    configurations = [
        (
            copy.deepcopy(initialized_meter_trump),
            "cache/tokens_trump.pkl",
            "../lexicon/ANEW2017/ANEW2017All.txt",
            "output_trump_anew.csv",
            "tokens",
            microsoft_translator_api_key
        ),
        (
            copy.deepcopy(initialized_meter_trump),
            "cache/tokens_trump.pkl",
            "../lexicon/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt",
            "output_trump_nrc.csv",
            "tokens",
            microsoft_translator_api_key
        ),
        (
            copy.deepcopy(initialized_meter),
            "cache/tokens.pkl",
            "../lexicon/ANEW2017/ANEW2017All.txt",
            "output_anew.csv",
            "tokens",
            microsoft_translator_api_key
        ),
        (
            copy.deepcopy(initialized_meter),
            "cache/tokens.pkl",
            "../lexicon/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt",
            "output_nrc.csv",
            "tokens",
            microsoft_translator_api_key
        )
    ]

    # ============
    # Execute in parallel
    # ============

    freeze_support()
    pool = Pool(processes=len(configurations))
    pool.starmap(launch, configurations)
