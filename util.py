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

# parsing a Roman Numeral symbol
class RomanNumeral:
    # TODO: NEED TO CHANGE ENCODING TO INCORPORATE GOLBALKEY INFO SO TAHT THIS IS WELL-DEFINED.
    # instances: 'II', 'bIII', 'bbbIV', '#II/VI'
    def __init__(self, roman_numeral: str):
        self.the_string = roman_numeral
        self.MAJOR_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
        self.MINOR_NUMERALS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
        self.ARABIC_NUMERALS_DICT = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6', 'VII': '7',
                                     'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6', 'vii': '7'}
        self.Accidental_RN = self._split_accidental_RN()
        self.quality = self._determine_quality()
        self.accidental = self._determine_accidental()
        # self.arabic_numeral = self._transform_to_arabic_numeral()
        # self.Accidental_AN = self._join_accidental_AN()

    def _split_accidental_RN(self):
        """Transform a RN str (with/without) accidental ('##II', 'II')
        to a list of two elements (e.g., ['##', 'II'] or ['', 'II]')"""

        accidental_count_is_1 = any([self.the_string.count('#') == 1, self.the_string.count('b') == 1])
        accidental_count_more = any([self.the_string.count('#') > 1, self.the_string.count('b') > 1])
        accidental_count_is_0 = all([[self.the_string.count('#') == 0, self.the_string.count('b') == 0]])

        seperated_str = re.split(r'([#b])', self.the_string)
        seperated_str = [x for x in seperated_str if x]  # remove emtpy str in list

        if accidental_count_is_1:
            return seperated_str

        elif accidental_count_more:
            accidental_part = ''.join(seperated_str[0:-1])
            seperated_rn = [accidental_part,seperated_str[-1]]
            return seperated_rn

        elif accidental_count_is_0:
            seperated_str = [self.the_string]
            seperated_str.insert(0, '')
            return seperated_str

    def _determine_quality(self) -> Literal['M', 'm']:
        """Given a split_accidental_RN (e.g., ['bb', 'III']), determine its quality (e.g., as 'M')"""

        the_rn = self._split_accidental_RN()[1]

        if any((c in self.MAJOR_NUMERALS) for c in the_rn):
            return 'M'
        elif any((c in self.MINOR_NUMERALS) for c in the_rn):
            return 'm'
        else:
            raise ValueError(f'Unexpected mode quality.')

    def _determine_accidental(self):
        """Given a roman numeral str (e.g., '#II'), determine its accidental (e.g., as '#')"""
        the_accidental = self._split_accidental_RN()[0]

        if any((c in ['#']) for c in the_accidental):
            return '#'
        elif any((c in ['b']) for c in the_accidental):
            return 'b'
        else:
            return the_accidental

    def _transform_to_arabic_numeral(self):
        """Given a roman numeral str (e.g., '#II'), determine its arabic numeral part (e.g., as '2')
        Some edgecase like 'bIII/VI', 'V/V/V'
        """
        roman_numeral_part = self._split_accidental_RN()[1]
        if '/' in roman_numeral_part:
            seperated_rn = roman_numeral_part.split('/')
            for idx, rn in enumerate(seperated_rn):
                abs_arabic_numeral = self.ARABIC_NUMERALS_DICT[rn]
                print(abs_arabic_numeral)
                # TODO: current version cannot account for cases like 'V/V/V'
            return 'great'
        else:
            arabic_numeral = self.ARABIC_NUMERALS_DICT[roman_numeral_part]
            return arabic_numeral

    def _join_accidental_AN(self):
        """Transform a roman numeral str with accidental (e.g. 'bvi') to arabic numeral with accidental(e.g.'b6')."""
        AN = self.arabic_numeral
        accidental = self._split_accidental_RN()[0]
        Accidental_AN = ''.join([accidental, AN])
        return Accidental_AN


def determine_modulation_bigram_type(modulation_bigram: str) -> Literal['MM', 'Mm', 'mM', 'mm']:
    preceding, following = modulation_bigram.split('_')[1:]
    preceding_numeral = RomanNumeral(preceding)
    following_numeral = RomanNumeral(following)
    bigram_type = preceding_numeral.quality + following_numeral.quality
    return bigram_type


def filter_modulation_bigrams_by_types(modulation_bigrams_list: List[str],
                                       modulation_type: Literal['MM', 'Mm', 'mM', 'mm']) -> List[str]:
    """Given a list of modulation bigrams, filter the list by modulation_type   """
    filtered_bigrams = [bigram for bigram in modulation_bigrams_list if
                        determine_modulation_bigram_type(bigram) == modulation_type]
    return filtered_bigrams


def compute_bigram_modulation_steps(bigram: str, fifths: bool = False):
    """Transform a bigram (e.g.'Db_I_#II') to a interval to scale degree (e.g.'Db_1_#2') and to
    finally calculate the modulation step (e.g.'a2'), if fifths is true, return (e.g. '')"""

    # roman numeral unit --> arabic numeral unit
    globalkey = bigram.split('_')[0]
    preceding, following = bigram.split('_')[1:]
    preceding_numeral = RomanNumeral(preceding)
    following_numeral = RomanNumeral(following)

    preceding_numeral_arabic = preceding_numeral.Accidental_AN
    following_numeral_arabic = following_numeral.Accidental_AN

    arabic_numeral_unit = '_'.join([globalkey, preceding_numeral_arabic, following_numeral_arabic])

    # encode scale degree (arabic numeral) to spelled pitch (transposed to C) with ref to globalkey
    MAJOR_SCALE_DEGREE_DICT = {'1': 'C', '2': 'D', '3': 'E', '4': 'F', '5': 'G', '6': 'A', '7': 'B',
                               '#1': 'C#', '#2': 'D#', '#3': 'E#', '#4': 'F#', '#5': 'G#', '#6': 'A#', '#7': 'B#',
                               'b1': 'Cb', 'b2': 'Db', 'b3': 'Eb', 'b4': 'Fb', 'b5': 'Gb', 'b6': 'Ab', 'b7': 'Bb'}
    MINOR_SCALE_DEGREE_DICT = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E', '6': 'F', '7': 'G',
                               '#1': 'A#', '#2': 'B#', '#3': 'C#', '#4': 'D#', '#5': 'E#', '#6': 'F#', '#7': 'G#',
                               'b1': 'Ab', 'b2': 'Bb', 'b3': 'Cb', 'b4': 'Db', 'b5': 'Eb', 'b6': 'Fb', 'b7': 'Gb'}

    # arabic numeral unit --> spelled pitch (2 categories: Major and Minor key mode)
    key = arabic_numeral_unit.split('_')[0][0]  # take the 'D' in 'Db'
    mode = 'M' if key.isupper() else 'm'
    preceding_sd, following_sd = arabic_numeral_unit.split('_')[1:]

    if mode == 'M':
        preceding_SP = pitchtypes.SpelledPitchClass(MAJOR_SCALE_DEGREE_DICT[preceding_sd])
        following_SP = pitchtypes.SpelledPitchClass(MAJOR_SCALE_DEGREE_DICT[following_sd])
        interval = following_SP - preceding_SP

    else:
        preceding_SP = pitchtypes.SpelledPitchClass(MINOR_SCALE_DEGREE_DICT[preceding_sd])
        following_SP = pitchtypes.SpelledPitchClass(MINOR_SCALE_DEGREE_DICT[following_sd])
        interval = following_SP - preceding_SP

    if fifths:
        interval_in_fifths = interval.fifths()
        return interval_in_fifths
    else:
        return interval


def compute_modulation_steps(bigrams_list: List[str], fifths: bool = True):
    modulation_steps_list = []
    for item in bigrams_list:
        step = compute_bigram_modulation_steps(item, fifths=fifths)
        modulation_steps_list.append(step)
    return modulation_steps_list


if __name__ == '__main__':
    localkey_bigrams = ['F_I_V', 'F_V_V', 'F_V_vi', 'F_vi_I', 'F_I_V', 'F_vi_i']
    bigrams = ['Db_I_#II', 'Db_#II_vi', 'f#_i_bIII', 'f#_bIII_i']

    # rn = 'bVI'
    # print(RomanNumeral(rn).Accidental_AN)

    # bigram = 'f#_i_bbIII'
    # double_flat_rn = 'f#_i_bbIII'
    # double_mod_rn = 'a_VI_bIII/VI'

    split_acc_rn_test = 'V/V/V'
    split_acc_rn_test = 'bbbIII'

    result = RomanNumeral(split_acc_rn_test)._transform_to_arabic_numeral()
    print(result)

