#  By Xinyi Guan on 25 November 2022.
from typing import List

import numpy as np
import pandas as pd


# ===================================
# n-gram                            |
# ===================================

def get_n_grams(sequence: List[str], n: int) -> np.ndarray:
    """
    Transform a list of string to a list of n-grams.
    :param sequence:
    :param n:
    :return:
    """
    transitions = np.array(
        [['_'.join(sequence[i - (n - 1):i]), sequence[i]] for i in range(n - 1, len(sequence))])
    return transitions


def get_transition_matrix(n_grams: np.ndarray) -> pd.DataFrame:
    """
    Transform the n-gram np-array to a transition matrix dataframe
    :param n_grams:
    :return:
    """
    contexts, targets = np.unique(n_grams[:, 0]), np.unique(n_grams[:, 1])
    transition_matrix = pd.DataFrame(0, columns=targets, index=contexts)
    for i, n_gram in enumerate(n_grams):
        context, target = n_gram[0], n_gram[1]
        transition_matrix.loc[context, target] += 1
        # print(transition_matrix)
    return transition_matrix


def determine_era_based_on_year(year) -> str:
    if 0 < year < 1650:
        return 'Renaissance'

    elif 1649 < year < 1759:
        return 'Baroque'

    elif 1758 < year < 1819:
        return 'Classical'

    elif 1819 < year < 1931:
        return 'Romantic'


MAJOR_MINOR_KEYS_Dict = {'A': 'major', 'B': 'major', 'C': 'major', 'D': 'major', 'E': 'major', 'F': 'major',
                         'G': 'major',
                         'A#': 'major', 'B#': 'major', 'C#': 'major', 'D#': 'major', 'E#': 'major', 'F#': 'major',
                         'G#': 'major',
                         'Ab': 'major', 'Bb': 'major', 'Cb': 'major', 'Db': 'major', 'Eb': 'major', 'Fb': 'major',
                         'Gb': 'major',
                         'a': 'minor', 'b': 'minor', 'c': 'minor', 'd': 'minor', 'e': 'minor',
                         'f': 'minor', 'g': 'minor',
                         'a#': 'minor', 'b#': 'minor', 'c#': 'minor', 'd#': 'minor', 'e#': 'minor', 'f#': 'minor',
                         'g#': 'minor',
                         'ab': 'minor', 'bb': 'minor', 'cb': 'minor', 'db': 'minor', 'eb': 'minor',
                         'fb': 'minor', 'gb': 'minor'}