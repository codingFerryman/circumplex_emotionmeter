# Load the name of the cities, states, and counties in the United States
import calendar
import re
import string

import inflect
import yaml
from yaml import Loader
from nltk.corpus import stopwords
from utils import yaml_load


def load_procedural_words():
    procedural_words = ['house', 'senate', 'congress', 'speaker', 'chairman', 'member', 'committee', 'gentleman',
                        'gentlelady', 'gentlemen', 'floor', 'senator', 'congressmen', 'congressman', 'congresswomen',
                        'congresswoman', 'yield', 'democrat', 'republican', 'chair', 'state']
    return procedural_words


def load_nltk_stopwords():
    """NLTK Stopwords"""
    nltk_stopwords = stopwords.words('english')
    return nltk_stopwords


def load_areas():
    """Load the name of the cities, states, and counties in the United States"""
    with open('../USA-cities-and-states/us_cities_states_counties.csv', 'r') as fp:
        us_areas_file = fp.read()
    us_areas = list(
        set(re.split('[\n| ]', us_areas_file)) - {'city', 'state short', 'state full', 'county', 'city alias', ''})
    return us_areas


def load_legislators():
    """Load the name of the current and historical legislators"""
    name_list = []
    for res_path in ['legislators-current.yaml', 'legislators-historical.yaml']:
        with open('../congress-legislators/' + res_path, 'r', encoding='utf-8') as fp:
            name_res = yaml.load(fp, Loader=Loader)
        name_res_list = [v['name'][key] for v in name_res for key in ['first', 'last']]
        name_list.extend(name_res_list)
    return name_list


def load_number_words(start=0, end=1000):
    """Generate the ordinal and cardinal numbers"""
    assert start < end
    inflect_eng = inflect.engine()
    numbers_1 = [inflect_eng.number_to_words(inflect_eng.ordinal(i)) for i in range(max(0, start), min(101, end))]
    numbers_2 = [inflect_eng.ordinal(i) for i in range(end)]
    numbers_3 = [inflect_eng.number_to_words(i) for i in range(max(0, start), min(101, end))]
    numbers = [*numbers_1, *numbers_2, *numbers_3]
    return numbers


def load_calendar_words():
    """Names of month and weekdays"""
    month_name = list(calendar.month_name)[1:]
    weekday_name = list(calendar.day_name)
    calendar_words = [*month_name, *weekday_name]
    return calendar_words


class StopwordsLoader:
    def __init__(self, resources="nltk,numbers,procedural", lower=False, return_regex=False, incl_punt=True):
        self.resources = resources.split(',')
        self.lower = lower
        self.return_regex = return_regex
        self.incl_punt = incl_punt

    def load(self):
        _words = []
        if 'nltk' in self.resources:
            _words.extend(load_nltk_stopwords())
        if 'areas' in self.resources:
            _words.extend(load_areas())
        if 'legislators' in self.resources:
            _words.extend(load_legislators())
        if 'calendar' in self.resources:
            _words.extend(load_calendar_words())
        if 'numbers' in self.resources:
            _words.extend(load_number_words(0, 1000))
        if 'procedural' in self.resources:
            _words.extend(load_procedural_words())
        _words = set(_words)
        if self.lower:
            _words = set(map(lambda x: x.lower(), _words))
        if self.incl_punt:
            _words = _words.union(string.punctuation+'…')
        if self.return_regex:
            _words = list(_words)
            return re.compile(r'\b(' + r'|'.join(_words) + r')\b\s?')
        else:
            return _words


if __name__ == '__main__':
    _test = load_calendar_words()
    pass