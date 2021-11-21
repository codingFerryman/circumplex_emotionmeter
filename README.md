This repository is for the project in the course [Building a Robot Judge: Data Science for Decision-Making](http://www.vvz.ethz.ch/lerneinheitPre.do?semkez=2021W&lerneinheitId=146397&lang=en)

This project is still in progress.

For both valence and arousal values, the maximum (positive) value is ```1```, the minimum (negative) value is ```-1```, the neutral value is ```0```.

## Setup
After cloning this repository, create or activate a virtual environment, then execute:
```bash
cd PATH-TO-REPO-DIR
bash ./setup_env.sh
```

## Execution
Analyze a single text:
```python3
python main.py text=TEXT lexicon=PATH-TO-LEXICON-FILE
```

Analyze a single text:
```python3
python main.py data=PATH-TO-DATA-FILE lexicon=PATH-TO-LEXICON-FILE output=PATH-FOR-OUTPUT-FILE
```

## Results
Some results are visulized in [emotionmeter_anew2017_viz.ipynb](./emotionmeter_anew2017_viz.ipynb)

## TODO
- Take contrast connectives into account
- Plot the results in circumplexes
