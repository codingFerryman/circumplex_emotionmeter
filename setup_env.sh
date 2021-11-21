#!/usr/bin/bash

# Specify the project path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Upgrade the building tools
pip install --upgrade pip setuptools wheel

# Install the requirements
pip install -r $SCRIPT_DIR/emotionmeter/requirements.txt

# Download NLTK stopwords and Spacy corpus
python -c "import nltk; nltk.download('stopwords')"
python -m spacy download "en_core_web_lg"
python -m spacy download "en_core_web_sm"
