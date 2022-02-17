import multiprocessing
import re
import string
from pathlib import Path
from typing import Optional, Union, List, Set

import nltk
import pandas as pd
import requests
from spacy.cli.init_pipeline import init_vectors_cli

from stopwords_loader import StopwordsLoader
from utils import get_logger, apply_calculation_rows
from gensim.models import Word2Vec
from translate import Translator
import fasttext

logger = get_logger("preprocessor", True)


class EmotionText(object):
    """
    Identify the properties of input text
    Reference:
        Hutto, Clayton, and Eric Gilbert.
        "Vader: A parsimonious rule-based model for sentiment analysis of social media text."
        In Proceedings of the International AAAI Conference on Web and Social Media, vol. 8, no. 1. 2014.
    """

    def __init__(self,
                 text,
                 tokenizer,
                 stopwords: Union[List, Set, re.Pattern] = None,
                 stopwords_cap: Union[List, Set, re.Pattern] = None,
                 ):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')

        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.replace("\r", "")

        self.text = text

        if stopwords is None:
            stopwords = []
        if stopwords_cap is None:
            stopwords_cap = []
        self.stopwords = stopwords
        self.stopwords_cap = stopwords_cap
        self.tokens, self.hashtags, self.mentions, self.rt = self._tokenize(tokenizer)

    def _tokenize(self, tokenizer):
        """
        If nltk_tweet_tokenizer:
            tokenize the sentence by TweetTokenizer from NLTK
        Else:
            Removes leading and trailing puncutation
            Leaves contractions and most emoticons
                Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        _rt_label = False
        _rt = None
        if len(self.text) > 4:
            if self.text[:4] == 'RT @':
                _rt_label = True
        _tokens = tokenizer.tokenize(self.text)
        _tokens = [_t.strip(str(set(string.punctuation).difference({'@', '#'}))) for _t in _tokens]
        _hashtags = [_t for _t in _tokens if len(_t) > 1 and _t[0] == '#' and not _t[1:].isnumeric()]
        _mentions = [_t for _t in _tokens if len(_t) > 1 and _t[0] == '@']

        if _rt_label is True:
            _rt = _mentions.pop(0)
            _tokens = _tokens[3:]

        _tokens = [_t.lower() for _t in _tokens if _t not in self.stopwords_cap.union(set(_hashtags + _mentions))]
        _tokens = [_t for _t in _tokens if _t not in self.stopwords and not _t.startswith('http')]

        if len(_hashtags) == 0:
            _hashtags = None
        else:
            _hashtags = ','.join(_hashtags)

        if len(_mentions) == 0:
            _mentions = None
        else:
            _mentions = ','.join(_mentions)

        _tokens = [_t for _t in _tokens if len(_t) > 0]

        return _tokens, _hashtags, _mentions, _rt


def convert2spacy(gensim_vector_path, save_dir):
    logger.info(f"Converting the Gensim Word2Vec model from {gensim_vector_path} to Spacy model ...")
    init_vectors_cli(lang='en',
                     vectors_loc=Path(gensim_vector_path),
                     output_dir=Path(save_dir),
                     prune=-1, truncate=0, mode='default', name=None, verbose=False, jsonl_loc=None)
    logger.info(f"Converted successfully! Please load the Spacy model from {save_dir}")


class CircumplexEmotionMeterPreprocessor:
    def __init__(
            self,
            data_path_or_df: Optional[str] = "data/tweets/ExtractedTweets.csv",
            text_column: Optional[str] = "Tweet",
            lang_id_model_path: Optional[str] = './lid.176.bin',
            use_tqdm=True
    ):
        self.tokenizer = None
        self.text_column = text_column
        self.data_path_or_df = data_path_or_df
        self.data_df = None

        self.stopwords = []
        self.stopwords_cap = []

        self.lang_id_model_path = lang_id_model_path

        self.use_tqdm = use_tqdm

        self.result_df = None

    def load_stopwords(self, stopwords_cap=None, stopwords=None):
        if stopwords_cap is None:
            # stopwords_cap = "areas"
            stopwords_cap = "legislators,areas"
        logger.debug(f"Loading stopwords: {stopwords_cap}")
        self.stopwords_cap = StopwordsLoader(stopwords_cap).load()

        if stopwords is None:
            stopwords = "nltk,numbers,procedural,calendar"
        logger.debug(f"Loading stopwords: {stopwords}")
        self.stopwords = StopwordsLoader(stopwords, lower=True).load()

        logger.debug(f"Loaded {len(self.stopwords)} lowercase stopwords")
        logger.debug(f"Loaded {len(self.stopwords_cap)} stopwords having capitalized letters")

        logger.info("Stopwords loaded")

    def load_tokenizer(self, tokenizer=None, **kwargs):
        if tokenizer is None:
            self.tokenizer = nltk.ToktokTokenizer()
        else:
            self.tokenizer = tokenizer(kwargs)

    def load_data(self, data_path_or_df=None, text_column: str = None):
        """
        Load data into a DataFrame
        :param data_path_or_df:
        :param text_column: the column name of text
        :return:
        """
        if data_path_or_df is not None:
            self.data_path_or_df = data_path_or_df
        if type(self.data_path_or_df) is pd.DataFrame:
            self.data_df = self.data_path_or_df
        else:
            self.data_df = pd.read_csv(self.data_path_or_df)

        if text_column is None:
            text_column = self.text_column
        assert (text_column in self.data_df.columns), f"df must have column {text_column}"
        logger.info('Data loaded')

    def _lang_detect_and_translate(self, lang_model):
        res = lang_model.predict(self.data_df.Tweet.str.replace('\n', '').to_list())
        self.data_df['lang'] = [r[0][9:] for r in res[0]]
        self.data_df['lang_prob'] = [r[0] for r in res[1]]
        self.data_df.loc[self.data_df.lang_prob < 0.4, "lang"] = "en"
        self.data_df.loc[self.data_df.Tweet.str.startswith('http'), "lang"] = "en"
        langset = set(self.data_df.lang) - {'en'}
        for l in langset:
            translator = Translator(provider='microsoft', from_lang='', to_lang='en',
                                    secret_access_key="8115e9cac318499d901915fbbb55d700")
            self.data_df.loc[
                self.data_df['lang'] == l,
                'Tweet'
            ] = self.data_df.loc[self.data_df['lang'] == l]['Tweet'].map(
                lambda x: translator.translate(x)
            )

    def preprocess(self, **kwargs):
        assert (self.tokenizer is not None), "Please load a tokenizer first"
        assert (self.data_df is not None), "Please load data first"
        if self.lang_id_model_path is not None:
            lang_model = fasttext.load_model(self.lang_id_model_path)
        else:
            lang_model = None
        self._lang_detect_and_translate(lang_model)
        preprocessed = apply_calculation_rows(self.data_df, self.text_preprocessing, self.use_tqdm, **kwargs)
        self.result_df = pd.DataFrame(preprocessed)
        return self.result_df

    def save_tokens(self, save_path="./tokens.pkl"):
        assert self.result_df is not None
        self.result_df.to_pickle(save_path)

    def load_tokens(self, load_path="./tokens.pkl"):
        self.result_df = pd.read_pickle(load_path)
        return self.result_df

    def save_word2vec(self, save_path="./w2v.txt"):
        cores = multiprocessing.cpu_count()

        assert self.result_df is not None
        tokens = self.result_df.tokens.to_list()
        logger.info("Training Word2Vec ...")
        model = Word2Vec(sentences=tokens, vector_size=300, window=8, epochs=100, workers=cores - 1)
        model_vectors = model.wv
        model_vectors.save_word2vec_format(save_path)
        logger.info(f"The KeyedVector in Word2Vec is saved to {save_path}")
        return model

    def text_preprocessing(self, row_dict):
        t = EmotionText(row_dict[self.text_column],
                        self.tokenizer, self.stopwords, self.stopwords_cap)
        row_dict.update({
            'tokens': t.tokens,
            'hashtags': t.hashtags,
            'mentions': t.mentions,
            'rt': t.rt,
        })
        return row_dict


def execute_preprocessor(
        token_path='./cache/tokens.pkl',
        w2v_path='./cache/w2v.kv',
        spacy_save_path='./cache/w2v',
        **kwargs
):
    for p in [token_path, w2v_path, spacy_save_path]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    proc = CircumplexEmotionMeterPreprocessor(**kwargs)
    if not Path(token_path).is_file():
        lang_id_model_path = kwargs.get('lang_id_model_path', None)
        if lang_id_model_path is not None:
            Path(lang_id_model_path).parent.mkdir(parents=True, exist_ok=True)
            if not Path(lang_id_model_path).is_file():
                model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
                r = requests.get(model_url, allow_redirects=True)
                open(lang_id_model_path, 'wb').write(r.content)
        proc.load_stopwords()
        proc.load_tokenizer()
        proc.load_data()
        proc.preprocess()
        proc.save_tokens(token_path)
    logger.info(f"Loading tokens from {token_path} ...")
    _ = proc.load_tokens(token_path)
    proc.save_word2vec(w2v_path)
    convert2spacy(
        gensim_vector_path=w2v_path,
        save_dir=spacy_save_path
    )


if __name__ == '__main__':
    configurations = [
        {
            "token_path": './cache/tokens.pkl',
            "w2v_path": './cache/w2v.txt',
            "spacy_save_path": './cache/w2v',
            "args": {
                "data_path_or_df": '../data/tweets/ExtractedTweets.csv',
                "text_column": 'Tweet',
                "lang_id_model_path": './lid.176.bin'
            }
        },
        {
            "token_path": './cache/tokens_trump.pkl',
            "w2v_path": './cache/w2v_trump.txt',
            "spacy_save_path": './cache/w2v_trump',
            "args": {
                "data_path_or_df": '../data/tweets/trump_archive.csv',
                "text_column": 'doc',
                "lang_id_model_path": None
            }
        }
    ]

    for cfg in configurations:
        execute_preprocessor(
            token_path=cfg["token_path"],
            w2v_path=cfg["w2v_path"],
            spacy_save_path=cfg["spacy_save_path"],
            **cfg["args"]
        )
