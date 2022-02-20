# Circumplex Emotion Analysis for Political Tweets
This project extends the research from Gennaro and Ash 
([GitHub](https://github.com/elliottash/emotionmeter)) 
for analyzing the emotion of political tweets by representing the emotionality in valence, arousal, and dominance.

This repository is an individual project
in the course [Building a Robot Judge: Data Science for Decision-Making](http://www.vvz.ethz.ch/lerneinheitPre.do?semkez=2021W&lerneinheitId=146397&lang=en),
please refer to [GitHub repo](https://github.com/codingFerryman/circumplex_emotionmeter) for any pdates.


## Setup
After cloning this repository, create or activate a virtual environment, then execute [setup_env.sh](setup_env.sh):
```bash
cd PATH-TO-REPO-DIR
bash ./setup_env.sh
```
The NRC-VAD lexicon will be downloaded automatically, but you may also want to [request ANEW lexicon](https://csea.phhp.ufl.edu/media/anewmessage.html) by yourself.

## Execution
Prepare the data for analysis (Internet access is required, may need ~1.5 hours) by executing [preprocessing.py](src/preprocessing.py):
```bash
cd PATH-TO-REPO-DIR/src
bash ./preprocessing.py
```

Calculate the emotion scores by executing [main.py](src/main.py):
```bash
cd PATH-TO-REPO-DIR/src
bash ./main.py
```
If the data contains non-English texts but the lexicon only has English words, 
it is recommended to translate them to English first.
You can also try Microsoft Translator API and pass your secret key here. Please refer to the code and 
[Microsoft Azure Cognitive Service](https://www.microsoft.com/en-us/translator/business/translator-api/)
for more details. 

## Results Visualization
Some results are visulized in [exploration.ipynb](src/exploration.ipynb), [visualization.ipynb](src/visualization.ipynb),
and [word_cloud.ipynb](src/word_cloud.ipynb).
