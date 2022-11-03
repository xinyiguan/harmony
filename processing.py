from typing import List, Literal

import numpy as np
import pandas as pd


# =============================
# n-gram                      |
# =============================

def get_n_grams(sequence: List[str], n: int) -> np.ndarray:
    transitions = np.array(
        [['_'.join(sequence[i - (n - 1):i]), sequence[i]] for i in range(n - 1, len(sequence))])
    return transitions


def get_transition_matrix(n_grams: np.ndarray) -> pd.DataFrame:
    contexts, targets = np.unique(n_grams[:, 0]), np.unique(n_grams[:, 1])
    transition_matrix = pd.DataFrame(0, columns=targets, index=contexts)
    for i, n_gram in enumerate(n_grams):
        context, target = n_gram[0], n_gram[1]
        transition_matrix.loc[context, target] += 1
        # print(transition_matrix)
    return transition_matrix

# =============================
# modulation                  |
# =============================

MAJOR_NUMERALS = Literal['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', '#I', '#II', '#III', '#IV', '#V', '#VI', '#VII',
                         'bI', 'bII', 'bIII', 'bIV', 'bV', 'bVI', 'bVII']

MINOR_NUMERALS = Literal['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', '#i', '#ii', '#iii', '#iv', '#v', '#vi', '#vii',
                         'bi', 'bii', 'biii', 'biv', 'bv', 'bvi', 'bvii']

def partition_modualtion_bigrams_by_types(localkey_bigrams_list: List[str], partition_types: Literal['MM', 'Mm', 'mM', 'mm']):
    if partition_types is 'MM':
        MM = []
        for idx, val in enumerate(localkey_bigrams_list):
            pass
