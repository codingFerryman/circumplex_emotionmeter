import pandas as pd
import spacy
from pathlib import Path
from typing import Optional, Union, Dict, List

from preprocessing import CircumplexEmotionMeterPreprocessor, EmotionText
from utils import get_logger, apply_calculation_rows

logger = get_logger("circumplex_emotionmeter", True)


class CircumplexEmotionMeter:
    def __init__(self,
                 data_path_or_df: Optional[str, pd.DataFrame, Path] = "data/tweets/smallExtractedTweets.csv",
                 text_column: Optional[str] = "Tweet",
                 corpus: Union[str, Path] = "en_core_web_lg",
                 lexicon_path: Optional[str, Path] = "lexicon/ANEW2017/ANEW2017All.txt",
                 affect_path: Union[str, Path] = "../lexicon/affect_list.txt",
                 cognition_path: Union[str, Path] = "../lexicon/cognition_list.txt",
                 use_tqdm: bool = False
                 ):
        """
        Initialize emotion meter
        :param data_path_or_df: the path or a pd.DataFrame of a dataset, can be loaded via load_tokens()
        :param text_column: the column name of text, can be set via set_text_column()
        :param corpus: the name of Scapy corpus or the path of a pretrained pipeline
        :param lexicon_path: the path of lexicon file, can be loaded via load_lexicon()
        :param affect_path: the path of affect dictionary
        :param cognition_path: the path of cognition dictionary
        :param use_tqdm: show progress bar or not
        """

        self.text_column = text_column

        self.cognition_path = cognition_path
        self.affect_path = affect_path

        self._load_cognition_and_cognition_word_lists()
        self.nlp = spacy.load(corpus)
        self.nlp_cognition = self.nlp(' '.join(self.cog))
        self.nlp_affect = self.nlp(' '.join(self.aff))
        self.nlp_vad_dict = {}

        self.data_path_or_df = data_path_or_df
        self.data_df = None

        self.lexicon_path = lexicon_path
        self.lexicon_df = None

        self.tokenizer = None

        self.result_df = None
        self.result_list = []

        self.use_tqdm = use_tqdm

    def set_text_column(self, text_column: str = 'Tweet'):
        logger.debug(f"text_column set to {text_column}")
        self.text_column = text_column

    def _load_cognition_and_cognition_word_lists(self):
        with open(self.cognition_path, "r") as f:
            cognition_list = f.readlines()
        self.cog = [word.strip() for word in cognition_list]

        with open(self.affect_path, "r") as f:
            affect_list = f.readlines()
        self.aff = [word.strip() for word in affect_list]

        logger.info("Cognition and affect words loaded")

    def load_tokens(self, data_path_or_df=None, text_column: str = None, **kwargs):
        """
        Load preprocessed tokens.
        If no preprocessed tokens is found, the preprocessor will be loaded automatically
        """
        if data_path_or_df is not None:
            self.data_path_or_df = data_path_or_df
        if type(self.data_path_or_df) is pd.DataFrame:
            self.data_df = self.data_path_or_df
        else:
            self.data_df = pd.read_pickle(self.data_path_or_df)

        if text_column is None:
            text_column = self.text_column
        assert (text_column in self.data_df.columns), f"df must have column {text_column}"

        if text_column != 'tokens':
            save_path = kwargs.get('save_path', './tokens.pkl')
            preproc = CircumplexEmotionMeterPreprocessor(
                data_path_or_df=self.data_df,
                text_column=text_column,
                **kwargs
            )
            preproc.load_stopwords()
            preproc.load_tokenizer()
            preproc.load_data()
            preproc.preprocess()
            preproc.save_tokens(save_path)

        logger.info('Tokens loaded')

    def load_lexicon(self,
                     path: Optional[str, Path] = None,
                     class_ratio: Union[int, float] = 0.4,
                     **kwargs):
        """
        Import lexicon data
        :param path: the path of the lexicon file
        :param class_ratio: the threshold for word classification in the lexicon
        :param kwargs: parameters passed to pd.read_csv()
        """
        if path is None:
            _path = self.lexicon_path
        else:
            _path = path
        if 'ANEW2017' in _path:
            kwargs['sep'] = '\t'
            kwargs['index_col'] = 0
            # rating_scale = 9
            rating_min = 1
            rating_max = 9
        elif 'NRC' in _path:
            kwargs['sep'] = '\t'
            kwargs['index_col'] = 0
            kwargs['names'] = ['Word', 'ValMn', 'AroMn', 'DomMn']
            # rating_scale = 1
            rating_min = 0
            rating_max = 1
        else:
            raise FileNotFoundError
        self.lexicon_df = pd.read_csv(_path, na_filter=False, **kwargs)
        self._classify_lexicon(rating_min, rating_max, class_ratio=class_ratio)
        logger.info(f'Lexicon loaded {_path}')

    def _classify_lexicon(self, min_score, max_score, class_ratio=0.4):
        """
        Used for classify the words -> negative, neutral, positive
        ... based on their VAD scores in a lexicon
        """
        column_mapping_dict = {
            'valence': 'ValMn',
            'arousal': 'AroMn',
            'dominance': 'DomMn'
        }
        result_dict = {}
        assert 0 < class_ratio <= 0.5, "The ratio of one class (class_ratio) should be within (0, 0.5]"
        neutral_lower_limit = min_score + (max_score + 1 - min_score) * class_ratio
        neutral_upper_limit = max_score - (max_score + 1 - min_score) * class_ratio
        for dim in column_mapping_dict.keys():
            col = column_mapping_dict[dim]
            _tmp_neg = self.lexicon_df[self.lexicon_df[col] <= neutral_lower_limit].index.to_list()
            _tmp_pos = self.lexicon_df[self.lexicon_df[col] >= neutral_upper_limit].index.to_list()
            try:
                result_dict[dim + '_neg'] = " ".join(_tmp_neg)
            except TypeError:
                print(_tmp_neg)
                raise
            try:
                result_dict[dim + '_pos'] = " ".join(_tmp_pos)
            except TypeError:
                print(_tmp_pos)
                raise
        for k, v in result_dict.items():
            self.nlp_vad_dict[k] = self.nlp(result_dict[k])
            logger.debug(f"Loaded: {k}")

    def calculate_score_text(self, text_input: Union[Dict, List, str]):
        """
        Sum the valence and arousal sources of each word then calculate the average as the result
        :param text_input: a dictionary which has preprocessed information (tokens at least)
            ... it also accepts a single text or a list of tokens
        :return: result
        """
        assert (self.lexicon_df is not None), "Please load the lexicon file first"

        if isinstance(text_input, list):
            tokens = text_input
            hashtags = mentions = rt = None
        elif isinstance(text_input, str):
            tokens = EmotionText(text_input).tokens
            hashtags = EmotionText(text_input).hashtags
            mentions = EmotionText(text_input).mentions
            rt = EmotionText(text_input).rt
        else:
            tokens = text_input['tokens']
            hashtags = text_input.get('hashtags', None)
            mentions = text_input.get('mentions', None)
            rt = text_input.get('rt', None)

        if len(tokens) == 0:
            # print('Empty string, return None')
            return

        dimensions = ['valence', 'arousal', 'dominance']

        tokens_nlp = self.nlp(" ".join(tokens))

        if tokens_nlp.vector_norm:
            cognition_score = self.nlp_cognition.similarity(tokens_nlp)
            affect_score = self.nlp_affect.similarity(tokens_nlp)
            dim_score_result = {}
            for dim in dimensions:
                _pos_tag = dim + '_pos'
                _neg_tag = dim + '_neg'
                dim_score = self.nlp_vad_dict[_pos_tag].similarity(tokens_nlp) - self.nlp_vad_dict[_neg_tag].similarity(
                    tokens_nlp)
                dim_score_result[dim] = ((dim_score + 1) / (cognition_score + 1)) - 1

            emotionality_old = ((affect_score + 1) / (cognition_score + 1)) - 1
            addition_result = {'affect': affect_score,
                               'cognition': cognition_score,
                               'emotionality': emotionality_old,
                               'hashtags': hashtags, 'mentions': mentions, 'rt': rt,
                               'tokens': ' '.join(tokens)}
            text_input.update(dim_score_result)
            text_input.update(addition_result)
            self.result_list.append(text_input)

        else:
            return

    def calculate_score(self, data_df=None):
        if data_df is None:
            data_df = self.data_df
        assert (data_df is not None), "Please load the dataset first"
        logger.info('Calculating scores ... ')
        apply_calculation_rows(data_df, self.calculate_score_text, self.use_tqdm)
        self.result_df = pd.DataFrame(self.result_list)
        return self.result_df

    def save_score(self, file_name_or_path='result.csv'):
        """
        Save the scores to a file
        :param file_name_or_path: The path of the file
        """
        _abs_path = Path(file_name_or_path).resolve()
        tmp = self.result_df[self.text_column].apply(self.text_save_fix)
        self.result_df[self.text_column] = tmp
        self.result_df.to_csv(Path(file_name_or_path), index=False)
        logger.info(f'Results are exported to {_abs_path}')

    @staticmethod
    def text_save_fix(text):
        # Line breaks have to be removed since they may cause compatible issues on calling read_csv()
        return text.replace('\n', ' ').replace('\r', '')

    def self_refresh(self):
        self.data_df = None
        self.data_path_or_df = None
        self.lexicon_path = None
        self.lexicon_df = None
        self.lexicon_words = None
        self.valence = None
        self.arousal = None
        self.dominance = None
        self.set_text_column('')
        logger.debug("Refreshed!")
