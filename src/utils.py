import coloredlogs
import logging
import numpy as np
import random
from tqdm.auto import tqdm


def apply_calculation_rows(df, func, use_tqdm, **kwargs):
    results = []
    df_dict_list = df.to_dict('records')
    if use_tqdm:
        df_dict_list = tqdm(df_dict_list)
    for _r in df_dict_list:
        results.append(func(_r, **kwargs))
    return results


def set_seed(seed: int = 2021):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)


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
