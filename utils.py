import random
from pathlib import Path
import numpy as np
# import torch
import logging
import coloredlogs
import rtyaml

def set_seed(seed: int = 2021):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def get_project_path() -> Path:
    """The function for getting the root directory of the project"""
    try:
        import git
        return git.Repo(Path(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    except NameError:
        return Path(__file__).parent.parent


def get_data_path() -> Path:
    return Path(get_project_path(), 'data')


def get_logger(name: str, debug=False):
    fmt = '[%(asctime)s] - %(name)s - {line:%(lineno)d} %(levelname)s - %(message)s'
    logger = logging.getLogger(name=name)
    if debug:
        logger.setLevel(logging.DEBUG)
        coloredlogs.install(fmt=fmt, level='DEBUG', logger=logger)
    else:
        logger.setLevel(logging.INFO)
        coloredlogs.install(fmt=fmt, level='INFO', logger=logger)
    return logger


def yaml_load(path, use_cache=True):
    # Copyright: https://github.com/unitedstates/congress-legislators/blob/main/scripts/utils.py

    # Loading YAML is ridiculously slow, so cache the YAML data
    # in a pickled file which loads much faster.

    # Check if the .pickle file exists and a hash stored inside it
    # matches the hash of the YAML file, and if so unpickle it.
    import pickle as pickle, os.path, hashlib
    h = hashlib.sha1(open(path, 'rb').read()).hexdigest()
    if use_cache and os.path.exists(path + ".pickle"):

        try:
            store = pickle.load(open(path + ".pickle", 'rb'))
            if store["hash"] == h:
                return store["data"]
        except EOFError:
            pass # bad .pickle file, pretend it doesn't exist

    # No cached pickled data exists, so load the YAML file.
    data = rtyaml.load(open(path))

    # Store in a pickled file for fast access later.
    pickle.dump({ "hash": h, "data": data }, open(path+".pickle", "wb"))

    return data