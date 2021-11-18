from emotionmeter.emotionmeter import EmotionMeter


class EmotionMeterANEW2017(EmotionMeter):
    def __init__(self,
                 data_path: str = "data/smallExtractedTweets.csv",
                 text_column: str = "Tweet",
                 corpus: str = "en_core_web_lg",
                 lexicon_path: str = "lexicon/anew2017/ANEW2017All.txt"
                 ):
        super(EmotionMeterANEW2017, self).__init__(data_path, text_column, corpus)

        self.lexicon_path = lexicon_path
        self.lexicon = None

    def load_lexicon(self, path=None):
        if path is None:
            path = self.lexicon_path
