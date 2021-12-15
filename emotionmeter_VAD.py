import re

from stopwords_loader import StopwordsLoader
from utils import get_logger
from emotionmeter.emotionmeter import EmotionMeter
import pandas as pd
import string
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
import preprocessor as p

tqdm.pandas()

logger = get_logger("circumplex_emotionmeter", True)


class EmotionText(object):
    """
    Identify the properties of input text
    Reference:
        Hutto, Clayton, and Eric Gilbert.
        "Vader: A parsimonious rule-based model for sentiment analysis of social media text."
        In Proceedings of the International AAAI Conference on Web and Social Media, vol. 8, no. 1. 2014.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.tokens = self._tokenize()

    @staticmethod
    def _strip_punc_if_word(token):
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)
        if len(stripped) <= 2:
            return token
        return stripped

    def _tokenize(self, nltk_tweet_tokenizer=True, **nltk_kwargs):
        """
        If nltk_tweet_tokenizer:
            tokenize the sentence by TweetTokenizer from NLTK
        Else:
            Removes leading and trailing puncutation
            Leaves contractions and most emoticons
                Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        if nltk_tweet_tokenizer:
            preserve_case = nltk_kwargs.get('preserve_case', False)
            strip_handles = nltk_kwargs.get('strip_handles', True)
            reduce_len = nltk_kwargs.get('reduce_len', True)
            tokenizer = TweetTokenizer(preserve_case=preserve_case, strip_handles=strip_handles, reduce_len=reduce_len)
            return tokenizer.tokenize(self.text)
        else:
            wes = self.text.split()
            stripped = list(map(self._strip_punc_if_word, wes))
            return stripped


class CircumplexEmotionMeter(EmotionMeter):
    def __init__(self,
                 data_path: str = "data/tweets/smallExtractedTweets.csv",
                 text_column: str = "Tweet",
                 corpus: str = "en_core_web_lg",
                 lexicon_path: str = "lexicon/ANEW2017/ANEW2017All.txt"
                 ):
        """
        Initialize emotion meter
        :param data_path: the path of dataset
        :param text_column: the column of text
        :param corpus: the name of Scapy corpus
        :param lexicon_path: the path of lexicon file
        """
        super(CircumplexEmotionMeter, self).__init__(data_path, text_column, corpus)

        self.data_path = data_path
        self.data_df = None

        self.lexicon_path = lexicon_path
        self.lexicon_df = None

        self.result_df = None

    def load_cognition_and_cognition_word_lists(self):
        pass

    def load_data(self, path=None, text_column: str = "Tweet"):
        """
        Load data into a DataFrame
        :param path: the path of the data file
        :param text_column: the column name of text
        :return:
        """
        if path is None:
            _path = self.data_path
        else:
            _path = path
        _data_df = pd.read_csv(_path)
        assert (text_column in _data_df.columns), f"df must have column {text_column}"
        self.data_df = _data_df
        logger.debug('Data is loaded')

    def load_lexicon(self,
                     path=None,
                     rating_scale=9,
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

        self.lexicon_df = pd.read_csv(_path, **kwargs)
        logger.debug('Lexicon is loaded')

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

        logger.debug('Sources are normalized')

    @staticmethod
    def preprocess_text(tweet, keep_hashtag_text: bool = False):
        stopwords_cap = StopwordsLoader("legislators,areas").load()
        stopwords_low = StopwordsLoader("nltk,numbers,procedural,calendar", lower=True).load()
        # stopwords = list({*stopwords_cap, *stopwords_low})
        if not keep_hashtag_text:
            p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)
        else:
            p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)

        tweet = re.sub('#', '', p.clean(tweet))
        tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
        tweet = " ".join(word for word in tweet.split() if word not in stopwords_cap)
        tweet = tweet.lower()
        tweet = " ".join(word for word in tweet.split() if word not in stopwords_low)
        return tweet

    def calculate_score_text(self, text):
        # TODO: Take contrast connectives into account
        """
        Sum the valence and arousal sources of each word then calculate the average as the result
        :param text: text for processing
        :return: result
        """
        assert (self.lexicon_df is not None), "Please load the lexicon file first"

        t = EmotionText(text)
        tokens = t.tokens

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

        return {'valence': valence, 'arousal': arousal, 'dominance': dominance}

    @staticmethod
    def _rescale_score(score):
        if score > 1:
            score = 1 - (1 / score)
        elif score < 1:
            score = score - 1
        else:
            score = 0
        return score

    def calculate_score(self, data_df=None):
        if data_df is None:
            data_df = self.data_df
        assert (data_df is not None), "Please load the dataset first"
        logger.info('Calculating scores ... ')
        _tmp_result = data_df[self.text_column].progress_apply(self.calculate_score_text)
        self.result_df = pd.concat([data_df, _tmp_result.apply(pd.Series)], axis=1)
        return self.result_df

    def save_score(self, file_name_or_path='valence_arousal.csv'):
        _abs_path = Path(file_name_or_path).resolve()
        self.result_df[self.text_column] = self.result_df[self.text_column].apply(self.text_save_fix)
        self.result_df.to_csv(Path('.', file_name_or_path), index=False)
        logger.info(f'Result is exported to {_abs_path}')

    @staticmethod
    def text_save_fix(text):
        # Line breaks have to be removed since they may cause compatible issues on calling read_csv()
        return text.replace('\n', ' ').replace('\r', '')