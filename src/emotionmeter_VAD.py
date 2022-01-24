import nltk
import numpy as np
# from emotionmeter.emotionmeter import EmotionMeter
import pandas as pd
import re
import spacy
import string
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Set

from stopwords_loader import StopwordsLoader
from utils import get_logger

# from tqdm.contrib.concurrent import process_map


logger = get_logger("circumplex_emotionmeter", True)


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
                 stopwords_cap: Union[List, Set, re.Pattern] = None):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
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

        return _tokens, _hashtags, _mentions, _rt


class CircumplexEmotionMeter():
    def __init__(self,
                 data_path_or_df: str = "data/tweets/smallExtractedTweets.csv",
                 text_column: str = "Tweet",
                 corpus: str = "en_core_web_lg",
                 lexicon_path: str = "lexicon/ANEW2017/ANEW2017All.txt",
                 # affection_path="../emotionmeter/word_lists/affect_list.txt",  # dummy argument
                 cognition_path="../lexicon/cognition_list.txt",
                 use_tqdm=False,
                 **kwargs):
        """
        Initialize emotion meter
        :param data_path: the path of dataset
        :param text_column: the column of text
        :param corpus: the name of Scapy corpus
        :param lexicon_path: the path of lexicon file
        """
        # super(CircumplexEmotionMeter, self).__init__(
        #     data_path_or_df, text_column,
        #     corpus="en_core_web_lg",
        #     affection_path=affection_path,  # dummy argument
        #     cognition_path=cognition_path)

        self.text_column = text_column
        # self.affection_path = affection_path
        self.cognition_path = cognition_path

        self.load_cognition_and_cognition_word_lists()
        self.nlp = spacy.load(corpus)
        self.nlp_cognition = self.nlp(' '.join(self.cog))

        self.data_path_or_df = data_path_or_df
        self.data_df = None

        self.lexicon_path = lexicon_path
        self.lexicon_df = None

        self.tokenizer = None

        self.result_df = None
        self.result_list = []

        self.use_tqdm = use_tqdm

    def load_cognition_and_cognition_word_lists(self):
        # with open(self.affection_path, "r") as f:
        #     affect_list = f.readlines()
        # self.aff = [word.strip() for word in affect_list]
        with open(self.cognition_path, "r") as f:
            cognition_list = f.readlines()
        self.cog = [word.strip() for word in cognition_list]

    def load_data(self, text_column: str = None):
        """
        Load data into a DataFrame
        :param text_column: the column name of text
        :return:
        """

        if type(self.data_path_or_df) is pd.DataFrame:
            self.data_df = self.data_path_or_df
        else:
            self.data_df = pd.read_csv(self.data_path_or_df)
        if text_column is None:
            text_column = self.text_column
        assert (text_column in self.data_df.columns), f"df must have column {text_column}"
        logger.info('Data loaded')

    def load_lexicon(self,
                     path=None,
                     valence_col='ValMn',
                     arousal_col='AroMn',
                     dominance_col='DomMn',
                     **kwargs):
        """
        Import lexicon data
        :param path: the path of the lexicon file
        :param rating_scale: the number of rating points
        :param valence_col: the name of valence column
        :param arousal_col: the name of arousal column
        :param dominance_col: the name of dominance column
        :param kwargs: parameters passed to pd.read_csv()
        :return:
        """
        if path is None:
            _path = self.lexicon_path
        else:
            _path = path
        if 'ANEW2017' in _path:
            kwargs['sep'] = '\t'
            kwargs['index_col'] = 0
            rating_scale = 9
        elif 'NRC' in _path:
            kwargs['sep'] = '\t'
            kwargs['index_col'] = 0
            kwargs['names'] = ['Word', 'ValMn', 'AroMn', 'DomMn']
            rating_scale = 1
        else:
            raise FileNotFoundError
        self.lexicon_df = pd.read_csv(_path, **kwargs)
        logger.info(f'Lexicon loaded {_path}')

        if rating_scale > 1:
            rating_neutral = int(0.5 + rating_scale / 2)
            norm_max_rating = rating_scale - rating_neutral
        elif rating_scale == 1:
            rating_neutral = 0.5
            norm_max_rating = 0.5
        else:
            raise ValueError

        self.lexicon_words = self.lexicon_df.index
        self.valence = (self.lexicon_df[valence_col] - rating_neutral) / norm_max_rating
        self.arousal = (self.lexicon_df[arousal_col] - rating_neutral) / norm_max_rating
        self.dominance = (self.lexicon_df[dominance_col] - rating_neutral) / norm_max_rating

        logger.debug('Scores are normalized to [-1, 1]')

    def load_stopwords(self):
        # stopwords_cap_setting = "areas"
        stopwords_cap_setting = "legislators,areas"
        logger.debug(f"Loading stopwords: {stopwords_cap_setting}")
        self.stopwords_cap = StopwordsLoader(stopwords_cap_setting).load()

        stopwords_setting = "nltk,numbers,procedural,calendar"
        logger.debug(f"Loading stopwords: {stopwords_setting}")
        self.stopwords = StopwordsLoader(stopwords_setting, lower=True).load()
        logger.info("Stopwords loaded")

    def load_tokenizer(self, tokenizer=None, **kwargs):
        if tokenizer is None:
            self.tokenizer = nltk.ToktokTokenizer()
        else:
            self.tokenizer = tokenizer(kwargs)

    @staticmethod
    def _rescale_score(score):
        if score > 1:
            score = 1 - (1 / score)
        elif score < 1:
            score = score - 1
        else:
            score = 0
        return score

    def calculate_score_text(self, row_dict):
        """
        Sum the valence and arousal sources of each word then calculate the average as the result
        :param row_dict: text for processing
        :return: result
        """
        assert (self.lexicon_df is not None), "Please load the lexicon file first"
        assert (self.tokenizer is not None), "Please load a tokenizer first"

        text = row_dict[self.text_column]
        t = EmotionText(text, self.tokenizer, self.stopwords, self.stopwords_cap)
        tokens, hashtags, mentions, rt = t.tokens, t.hashtags, t.mentions, t.rt

        if len(tokens) == 0:
            # print('Empty string, return None')
            return

        valence_list = [1]
        valence_neg_list = [1]
        arousal_list = [1]
        arousal_neg_list = [1]
        dominance_list = [1]
        dominance_neg_list = [1]

        for tk in tokens:
            if tk in self.lexicon_words:
                v = self.valence[tk]
                a = self.arousal[tk]
                d = self.dominance[tk]
                if v > 0:
                    valence_list.append(v)
                elif v < 0:
                    valence_neg_list.append(np.abs(v))
                if a > 0:
                    arousal_list.append(a)
                elif a < 0:
                    arousal_neg_list.append(np.abs(a))
                if d > 0:
                    dominance_list.append(d)
                elif d < 0:
                    dominance_neg_list.append(np.abs(d))

        valence_ratio = sum(valence_list) / sum(valence_neg_list)
        arousal_ratio = sum(arousal_list) / sum(arousal_neg_list)
        dominance_ratio = sum(dominance_list) / sum(dominance_neg_list)

        valence = self._rescale_score(valence_ratio)
        arousal = self._rescale_score(arousal_ratio)
        dominance = self._rescale_score(dominance_ratio)

        tokens_nlp = self.nlp(" ".join(tokens))

        if not tokens_nlp.vector_norm:
            cognition_score = self.nlp_cognition.similarity(tokens_nlp)
        else:
            cognition_score = 0

        if cognition_score != 0:
            valence_cog = valence / cognition_score
            arousal_cog = arousal / cognition_score
            dominance_cog = dominance / cognition_score
        else:
            valence_cog = arousal_cog = dominance_cog = None

        score_result = {'valence': valence, 'arousal': arousal, 'dominance': dominance,
                        'cognition': cognition_score,
                        'valence_cog': valence_cog, 'arousal_cog': arousal_cog, 'dominance_cog': dominance_cog,
                        'hashtags': hashtags, 'mentions': mentions, 'rt': rt,
                        'tokens': ' '.join(tokens)}
        row_dict.update(score_result)
        self.result_list.append(row_dict)

    def apply_calculation_rows(self, df):
        df_dict_list = df.to_dict('records')
        if self.use_tqdm:
            df_dict_list = tqdm(df_dict_list)
        for _r in df_dict_list:
            self.calculate_score_text(_r)

    def calculate_score(self, data_df=None):
        if data_df is None:
            data_df = self.data_df
        assert (data_df is not None), "Please load the dataset first"
        logger.info('Calculating scores ... ')
        self.apply_calculation_rows(data_df)
        self.result_df = pd.DataFrame(self.result_list)
        return self.result_df

    def save_score(self, file_name_or_path='valence_arousal.csv'):
        _abs_path = Path(file_name_or_path).resolve()
        tmp = self.result_df[self.text_column].apply(self.text_save_fix)
        self.result_df[self.text_column] = tmp
        self.result_df.to_csv(Path(file_name_or_path), index=False)
        logger.info(f'Results are exported to {_abs_path}')

    @staticmethod
    def text_save_fix(text):
        # Line breaks have to be removed since they may cause compatible issues on calling read_csv()
        return text.replace('\n', ' ').replace('\r', '')
