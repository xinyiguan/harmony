# =============================
# modulation                  |
# =============================

from typing import List, Literal

import numpy as np
import pandas as pd
from pitchtypes import AbstractBase
import pitchtypes

MAJOR_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', '#I', '#II', '#III', '#IV', '#V', '#VI', '#VII',
                  'bI', 'bII', 'bIII', 'bIV', 'bV', 'bVI', 'bVII']

MINOR_NUMERALS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', '#i', '#ii', '#iii', '#iv', '#v', '#vi', '#vii',
                  'bi', 'bii', 'biii', 'biv', 'bv', 'bvi', 'bvii']

MAJOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS = {'C': 'I',
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

MINOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS = {'C': 'i',
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


def partition_modualtion_bigrams_by_types(modulation_bigrams_list: List[str],
                                          partition_types: Literal['MM', 'Mm', 'mM', 'mm']) -> List[str]:
    """
    Partition the list of local_bigrams (e.g. 'F_I_iii') into a sub-list according to the partition type
    :param modulation_bigrams_list:
    :param partition_types:
    :return:
    """

    if partition_types == 'MM':
        MM = []
        for idx, val in enumerate(modulation_bigrams_list):
            preceding = val.split('_')[1]
            following = val.split('_')[2]
            if (preceding in MAJOR_NUMERALS) and (following in MAJOR_NUMERALS):
                MM.append(val)
        return MM

    elif partition_types == 'Mm':
        Mm = []
        for idx, val in enumerate(modulation_bigrams_list):
            preceding = val.split('_')[1]
            following = val.split('_')[2]
            if (preceding in MAJOR_NUMERALS) and (following in MINOR_NUMERALS):
                Mm.append(val)
        return Mm

    elif partition_types == 'mM':
        mM = []
        for idx, val in enumerate(modulation_bigrams_list):
            preceding = val.split('_')[1]
            following = val.split('_')[2]
            if (preceding in MINOR_NUMERALS) and (following in MAJOR_NUMERALS):
                mM.append(val)
        return mM

    elif partition_types == 'mm':
        mm = []
        for idx, val in enumerate(modulation_bigrams_list):
            preceding = val.split('_')[1]
            following = val.split('_')[2]
            if (preceding in MINOR_NUMERALS) and (following in MINOR_NUMERALS):
                mm.append(val)
        return mm


def compute_modulation_steps(partitioned_bigrams_list: List[str], partition_type: Literal['MM', 'Mm', 'mM', 'mm']):
    """
    Compute the modulation steps between the origin key and the target key.
    :param partitioned_bigrams_list:
    :param partition_type:
    :return:
    """

    def get_key(my_dict, val):
        for key, value in my_dict.items():
            if val == value:
                return key

    modulation_steps_list = []

    # using the Spelled Pitch in the pitchtype library
    mode_dict_mapping = {'M': MAJOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS,
                         'm': MINOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS}

    key_dict_1,key_dict_2 = (mode_dict_mapping.get(mode) for mode in partition_type)

    for idx, val in enumerate(partitioned_bigrams_list):
        preceding_RN = val.split('_')[1]
        following_RN = val.split('_')[2]

        preceding_SP = pitchtypes.SpelledPitchClass(
            get_key(my_dict=key_dict_1, val=preceding_RN))
        following_SP = pitchtypes.SpelledPitchClass(
            get_key(my_dict=key_dict_2, val=following_RN))

        interval = following_SP - preceding_SP
        modulation_steps_list.append(interval)
    return modulation_steps_list


if __name__ == '__main__':
    localkey_bigrams = ['F_I_V', 'F_V_V', 'F_V_vi', 'F_vi_I', 'F_I_V', 'F_vi_i']

    MM = ['F_I_I', 'F_I_V', 'F_V_I', 'F_V_V', 'F_bIII_I', 'F_I_bIII', 'F_I_III', 'F_I_#I', 'F_I_bIII']
    mM = ['F_vi_I', 'F_iv_I', 'F_vi_V', 'F_iv_V', 'F_iii_I']

    result = compute_modulation_steps(partitioned_bigrams_list=MM, partition_type='MM')

    print(result)

