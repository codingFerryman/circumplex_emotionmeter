from utils import get_logger
from emotionmeter.emotionmeter import EmotionMeter
import pandas as pd
import string
import numpy as np
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

logger = get_logger("emotionmeter_anew2017", True)

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

    def _tokenize(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped


class EmotionMeterANEW2017(EmotionMeter):
    def __init__(self,
                 data_path: str = "data/tweets/smallExtractedTweets.csv",
                 text_column: str = "Tweet",
                 corpus: str = "en_core_web_lg",
                 lexicon_path: str = "lexicon/anew2017/ANEW2017All.txt"
                 ):
        """
        Initialize emotion meter
        :param data_path: the path of dataset
        :param text_column: the column of text
        :param corpus: the name of Scapy corpus
        :param lexicon_path: the path of lexicon file
        """
        super(EmotionMeterANEW2017, self).__init__(data_path, text_column, corpus)

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
                     **kwargs):
        """
        Import lexicon data
        :param path: the path of the lexicon file
        :param rating_scale: the number of rating points
        :param valence_col: the name of valence column
        :param arousal_col: the name of arousal column
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

        rating_neutral = int(0.5 + rating_scale / 2)
        norm_max_rating = rating_scale - rating_neutral

        self.lexicon_words = self.lexicon_df.index
        self.valence = (self.lexicon_df[valence_col] - rating_neutral) / norm_max_rating
        self.arousal = (self.lexicon_df[arousal_col] - rating_neutral) / norm_max_rating

        logger.debug('Sources are normalized')

    def calculate_score_text(self, text):
        # TODO: Take contrast connectives into account
        """
        Sum the valence and arousal sources of each word then calculate the average as the result
        :param text: text for processing
        :return: result
        """
        assert (self.lexicon_df is not None), "Please load the lexicon file first"

        text = self.preprocess_text(text)
        t = EmotionText(text)
        tokens = t.tokens

        valence_list = [1]
        valence_neg_list = [1]
        arousal_list = [1]
        arousal_neg_list = [1]

        for tk in tokens:
            if tk in self.lexicon_words:
                v = self.valence[tk]
                a = self.arousal[tk]
                if v > 0:
                    valence_list.append(v)
                elif v < 0:
                    valence_neg_list.append(np.abs(v))
                if a > 0:
                    arousal_list.append(a)
                elif a < 0:
                    arousal_neg_list.append(np.abs(a))

        valence_ratio = sum(valence_list) / sum(valence_neg_list)
        arousal_ratio = sum(arousal_list) / sum(arousal_neg_list)

        valence = self._rescale_score(valence_ratio)
        arousal = self._rescale_score(arousal_ratio)

        return {'valence': valence, 'arousal': arousal, 'valence_ratio': valence_ratio, 'arousal_ratio': arousal_ratio}

    @staticmethod
    def _rescale_score(score):
        if score > 1:
            score = 1 - (1 / score)
        elif score < 1:
            score = score - 1
        else:
            score = 0
        return score

    def calculate_score(self):
        assert (self.data_df is not None), "Please load the dataset first"
        logger.info('Calculating scores ... ')
        _tmp_result = self.data_df[self.text_column].progress_apply(self.calculate_score_text)
        self.result_df = pd.concat([self.data_df, _tmp_result.apply(pd.Series)], axis=1)
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
