# Created by Xinyi Guan in 2022.

from typing import List, Literal, Union
import pitchtypes, re
import numpy as np
import pandas as pd
from scipy.stats import entropy


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


# ===================================
# modulation                        |
# ===================================

MAJOR_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII',
                  '#I', '#II', '#III', '#IV', '#V', '#VI', '#VII',
                  'bI', 'bII', 'bIII', 'bIV', 'bV', 'bVI', 'bVII']

MINOR_NUMERALS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii',
                  '#i', '#ii', '#iii', '#iv', '#v', '#vi', '#vii',
                  'bi', 'bii', 'biii', 'biv', 'bv', 'bvi', 'bvii']

MAJOR_MODE_RN_TO_SP = {'C': 'I',
                       'C#': '#I',
                       'Cb': 'bI',

                       'D': 'II',
                       'D#': '#II',
                       'Db': 'bII',

                       'E': 'III',
                       'E#': '#III',
                       'Eb': 'bIII',

                                              'F': 'IV',
                                              'F#': '#IV',
                                              'Fb': 'bIV',

                                              'G': 'V',
                                              'G#': '#V',
                                              'Gb': 'bV',

                                              'A': 'VI',
                       'A#': '#VI',
                       'Ab': 'bVI',

                       'B': 'VII',
                       'B#': '#VII',
                       'Bb': 'bVII'}

MINOR_MODE_RN_TO_SP = {'C': 'i',
                       'C#': '#i',
                       'Cb': 'bi',

                       'D': 'ii',
                       'D#': '#ii',
                       'Db': 'bii',

                       'E': 'iii',
                       'E#': '#iii',
                       'Eb': 'biii',

                                              'F': 'iv',
                                              'F#': '#iv',
                                              'Fb': 'biv',

                                              'G': 'v',
                                              'G#': '#v',
                       'Gb': 'bv',

                       'A': 'vi',
                       'A#': '#vi',
                       'Ab': 'bvi',

                       'B': 'vii',
                       'B#': '#vii',
                       'Bb': 'bvii'}


# parsing a Roman Numeral symbol
class RomanNumeral:
    def __init__(self, roman_numeral: str):
        self.the_string = roman_numeral
        self.quality = self._determine_quality()
        self.accidental = self._determine_accidental()

    def _determine_quality(self) -> Literal['M', 'm']:
        "Given a roman numeral (e.g., '#II'), determine its quality as 'M'"
        # use re

    def _determine_accidental(self) -> Literal['#', 'b']:
        pass


def determine_modulation_bigram_type(modulation_bigram: str) -> Literal['MM', 'Mm', 'mM', 'mm']:
    preceding, following = modulation_bigram.split('_')[1:]
    preceding_numeral = RomanNumeral(preceding)
    following_numeral = RomanNumeral(following)
    bigram_type = preceding_numeral.quality + following_numeral.quality
    return bigram_type


def filter_modulation_bigrams_by_types(modulation_bigrams_list: List[str],
                                       modulation_type: Literal['MM', 'Mm', 'mM', 'mm']) -> List[str]:
    """
    Filter the list of local_bigrams (e.g. 'F_I_iii') into a sub-list according to the partition type
    :param modulation_bigrams_list:
    :param partition_types:
    :return:
    """
    filtered_bigrams = [bigram for bigram in modulation_bigrams_list if
                        determine_modulation_bigram_type(bigram) == modulation_type]
    return filtered_bigrams


def compute_modulation_steps(bigrams_list: List[str],
                             ):
    """
    Transfrom the modulation bigram list (e.g., ['Db_I_#II', 'Db_#II_vi']) to scale degree list (e.g., ['Db_1_#2', ...])
    to finally calculate the modulation step (interval between spelled pitches) using the pitchtypes library.
    """

    # map all lowercase RN to uppercase in the bigram_list as the RN denote the scale degree
    capitalized_bigrams_list = []
    for idx, item in enumerate(bigrams_list):
        capitalized_str = item.replace('i', 'I').replace('v', 'V')
        capitalized_bigrams_list.append(capitalized_str)
    return capitalized_bigrams_list




if __name__ == '__main__':
    localkey_bigrams = ['F_I_V', 'F_V_V', 'F_V_vi', 'F_vi_I', 'F_I_V', 'F_vi_i']
    bigrams = ['Db_I_#II', 'Db_#II_vi']

    result = compute_modulation_steps(bigrams)
    # result2 = compute_modulation_steps(localkey_bigrams)

    print(result)
    # print(result2)
