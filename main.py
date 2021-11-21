from emotionmeter_anew2017 import EmotionMeterANEW2017
from utils import get_logger

logger = get_logger('main', True)


# sample_text = 'RT @garywhite13: @SenBillNelson will join @RepDarrenSoto at town hall Thursday in Haines City to discuss civil rights, restoration of votin…'
meter_test = EmotionMeterANEW2017(data_path="data/tweets/ExtractedTweets.csv")
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
meter_test.save_score(file_name_or_path='tweets_valence_arousal.csv')
