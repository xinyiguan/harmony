# Created by Xinyi Guan in 2022.
from dataclasses import dataclass
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
    # instances: 'II', 'bIII', 'bbbIV', '#II/VI'
    def __init__(self, roman_numeral: str):
        self.the_string = roman_numeral
        self.MAJOR_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
        self.MINOR_NUMERALS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
        self.Simple_ARABIC_NUMERAL_DICT = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6', 'VII': '7',
                                           'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6', 'vii': '7'}

        self.Secondary_ROMAN2ARABIC_NUMERALS_DICT = {'I': '0', 'II': '1', 'III': '2', 'IV': '3', 'V': '4', 'VI': '5',
                                                     'VII': '6',
                                                     'i': '0', 'ii': '1', 'iii': '2', 'iv': '3', 'v': '4', 'vi': '5',
                                                     'vii': '6'}

        self.Secondary_ARABIC_NUMERAL_DICT = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7'}

    def _split_accidental_RN(self):
        """Transform a RN str (with/without) accidental ('##II', 'II')
        to a list of two elements (e.g., ['##', 'II'] or ['', 'II]') """

        accidental_count_is_1 = any([self.the_string.count('#') == 1, self.the_string.count('b') == 1])
        accidental_count_more = any([self.the_string.count('#') > 1, self.the_string.count('b') > 1])
        accidental_count_is_0 = all([[self.the_string.count('#') == 0, self.the_string.count('b') == 0]])

        seperated_str = re.split(r'([#b])', self.the_string)
        seperated_str = [x for x in seperated_str if x]  # remove emtpy str in list

        if accidental_count_is_1:
            return seperated_str

        elif accidental_count_more:
            accidental_part = ''.join(seperated_str[0:-1])
            seperated_rn = [accidental_part, seperated_str[-1]]
            return seperated_rn

        elif accidental_count_is_0:
            seperated_str = [self.the_string]
            seperated_str.insert(0, '')
            return seperated_str

    def determine_quality(self) -> Literal['M', 'm']:
        """Given a split_accidental_RN (e.g., ['bb', 'III']), determine its quality (e.g., as 'M')
        Some edgecase like 'III/#VI'"""

        the_rn = self._split_accidental_RN()[1]
        if any((c in self.MAJOR_NUMERALS) for c in the_rn):
            return 'M'
        elif any((c in self.MINOR_NUMERALS) for c in the_rn):
            return 'm'
        else:
            raise ValueError(f'Unexpected mode quality.')

    def determine_accidental(self):
        """Given a roman numeral str (e.g., '#II'), determine its accidental (e.g., as '#')"""
        the_accidental = self._split_accidental_RN()[0]

        if any((c in ['#']) for c in the_accidental):
            return '#'
        elif any((c in ['b']) for c in the_accidental):
            return 'b'
        else:
            return the_accidental

    @staticmethod
    def _drop_accidentals(my_string):
        roman_numeral_part = my_string.replace('#', '').replace('b', '')
        return roman_numeral_part

    def transform_to_arabic_numeral(self):
        """Given a roman numeral str (e.g., '#II'), determine its arabic numeral part (e.g., as '#2')
        Some edgecase like 'bIII/VI' -> '', 'V/V/V'
        """
        roman_numeral_part = self._drop_accidentals(self.the_string)

        # account for secondary chord
        if '/' in roman_numeral_part:
            seperated_rn = roman_numeral_part.split('/')
            abs_arabic_numeral = sum(int(self.Secondary_ROMAN2ARABIC_NUMERALS_DICT[rn]) for rn in seperated_rn)
            intermediate_roman_arabic_numeral = (abs_arabic_numeral % 7)
            arabic_numeral = self.the_string.replace(roman_numeral_part, self.Secondary_ARABIC_NUMERAL_DICT[
                intermediate_roman_arabic_numeral])
            return arabic_numeral
        else:
            arabic_numeral = self.the_string.replace(roman_numeral_part,
                                                     self.Simple_ARABIC_NUMERAL_DICT[roman_numeral_part])
            return arabic_numeral

    def join_accidental_AN(self):
        """Transform a roman numeral str with accidental (e.g. 'bvi') to arabic numeral with accidental(e.g.'b6')."""
        AN = str(self.transform_to_arabic_numeral())
        accidental = self._split_accidental_RN()[0]
        Accidental_AN = ''.join([accidental, AN])
        return Accidental_AN


@dataclass
class ModulaitonBigram:
    # instances: 'C_bIII/V_ii/bIII/V'
    modulation_bigram: str

    def __post_init__(self):
        self.globalkey = self.modulation_bigram.split('_')[0]
        self.preceding_key = self.modulation_bigram.split('_')[1]
        self.following_key = self.modulation_bigram.split('_')[2]
        self.Simple_ARABIC_NUMERAL_DICT = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6', 'VII': '7',
                                           'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6', 'vii': '7'}
        self.scales_dict = self.read_scales_txt()

    @staticmethod
    def read_scales_txt():
        """read the scale.txt file """
        with open("scales.txt") as f:
            scales_dict = {str(key): value for line in f for (key, value) in [line.strip().split(':')]}
            for key, val in scales_dict.items():
                val = scales_dict[key]
                val_in_list = val.split(',')
                scales_dict[key] = val_in_list

        return scales_dict

    def _get_single_secondary_chord_pc(self, global_key, secondary_chord):
        """Transfrom secondary chord 'bIII/V' in C to its absolute spelled pc, e.g., Bb"""
        seperated_items = secondary_chord.split('/')
        preceding = seperated_items[0]
        following = seperated_items[1]

        preceding_numeral = preceding.replace('#', '').replace('b', '')
        following_numeral = following.replace('#', '').replace('b', '')

        preceding_idx_in_scale = int(self.Simple_ARABIC_NUMERAL_DICT[preceding_numeral]) - 1
        following_idx_in_scale = int(self.Simple_ARABIC_NUMERAL_DICT[following_numeral]) - 1

        localkey_abs_pc = following.replace(following_numeral, self.scales_dict[global_key][preceding_idx_in_scale])
        func_abs_pc = preceding.replace(preceding_numeral, self.scales_dict[localkey_abs_pc][following_idx_in_scale])
        return func_abs_pc

    def get_preceding_abs_pc(self):
        # account for secondary chord, all relative to CMajor or a minor
        if '/' in self.preceding_key:
            seperated_items = self.preceding_key.split('/')
            preceding_abs_pc = []

            return preceding_abs_pc


def determine_modulation_bigram_type(modulation_bigram: str) -> Literal['MM', 'Mm', 'mM', 'mm']:
    preceding, following = modulation_bigram.split('_')[1:]

    # check if it's a secondary chord.
    if '/' in preceding:
        active_preceding = preceding.split('/')[0]
        preceding_numeral = RomanNumeral(active_preceding)
    else:
        preceding_numeral = RomanNumeral(preceding)

    if '/' in following:
        active_following = following.split('/')[0]
        following_numeral = RomanNumeral(active_following)
    else:
        following_numeral = RomanNumeral(following)
    bigram_type = preceding_numeral.determine_quality() + following_numeral.determine_quality()
    return bigram_type


def filter_modulation_bigrams_by_types(modulation_bigrams_list: List[str],
                                       modulation_type: Literal['MM', 'Mm', 'mM', 'mm']) -> List[str]:
    """Given a list of modulation bigrams, filter the list by modulation_type   """
    filtered_bigrams = [bigram for bigram in modulation_bigrams_list if
                        determine_modulation_bigram_type(bigram) == modulation_type]
    return filtered_bigrams


def compute_bigram_modulation_steps(bigram: str, fifths: bool = False):
    """Transform a bigram (e.g.'Db_I_#II') to an interval to scale degree (e.g.'Db_1_#2') and to
    finally calculate the modulation step (e.g.'a2'), if fifths is true, return (e.g. '')"""

    # roman numeral unit (#I, I)--> arabic numeral unit (#1, 1)
    globalkey = bigram.split('_')[0]
    preceding, following = bigram.split('_')[1:]
    preceding_numeral_part = RomanNumeral(preceding)
    following_numeral_part = RomanNumeral(following)
    print('preceding, following: ', preceding, following)

    preceding_numeral_arabic = preceding_numeral_part.transform_to_arabic_numeral()
    following_numeral_arabic = following_numeral_part.transform_to_arabic_numeral()

    arabic_numeral_unit = '_'.join([globalkey, preceding_numeral_arabic, following_numeral_arabic])
    print('arabic_numeral_unit: ', arabic_numeral_unit)

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
    print(preceding_sd, following_sd)

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
    bigrams = ['Db_I_#II', 'Db_#II_vi', 'f#_i_bIII', 'f#_bIII/V_ii/V']

    bigram = 'Ab_bIII/V_vi/V'

    # symbol = 'bIII/V'
    result = ModulaitonBigram(bigram).get_preceding_abs_pc()

    print(result)
