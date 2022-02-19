#!/usr/bin/bash

# Specify the project path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Pull and update submodules
git submodule update --init --recursive
git submodule update --remote --merge

# Upgrade the building tools
pip install --upgrade pip setuptools

# Install the requirements
pip install -r $SCRIPT_DIR/requirements.txt

# Download NLTK stopwords and Spacy corpus
python -c "import nltk; nltk.download('stopwords')"
python -m spacy download "en_core_web_lg"
# python -m spacy download "en_core_web_sm"

# Download data files (please refer to the descriptions at https://github.com/elliottash/emotionmeter)
wget "https://polybox.ethz.ch/index.php/s/Us2HeNYzsu509dm/download?path=%2Fdata&files=ExtractedTweets.csv" -O data/tweets/ExtractedTweets.csv
wget "https://polybox.ethz.ch/index.php/s/Us2HeNYzsu509dm/download?path=%2Fdata&files=trump_archive.csv" -O data/tweets/trump_archive.csv

# Download lexicons
## For ANEW lexicon, please send a request at: https://csea.phhp.ufl.edu/media/anewmessage.html
## NRC-VAD lexicon
wget https://saifmohammad.com/WebDocs/VAD/NRC-VAD-Lexicon-Aug2018Release.zip -O .temp.zip
unzip .temp.zip -j lexicon/
rm .temp.zip
