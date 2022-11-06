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
                                              'Fb': ['bIV'],

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





def partition_modualtion_bigrams_by_types(localkey_bigrams_list: List[str],
                                          partition_types: Literal['MM', 'Mm', 'mM', 'mm']) -> List[str]:
    if partition_types == 'MM':
        MM = []
        for idx, val in enumerate(localkey_bigrams_list):
            preceding = val.split('_')[0]
            following = val.split('_')[1]
            if (preceding in MAJOR_NUMERALS) and (following in MAJOR_NUMERALS):
                MM.append(val)
        return MM

    elif partition_types == 'Mm':
        Mm = []
        for idx, val in enumerate(localkey_bigrams_list):
            preceding = val.split('_')[0]
            following = val.split('_')[1]
            if (preceding in MAJOR_NUMERALS) and (following in MINOR_NUMERALS):
                Mm.append(val)
        return Mm

    elif partition_types == 'mM':
        mM = []
        for idx, val in enumerate(localkey_bigrams_list):
            preceding = val.split('_')[0]
            following = val.split('_')[1]
            if (preceding in MINOR_NUMERALS) and (following in MAJOR_NUMERALS):
                mM.append(val)
        return mM

    elif partition_types == 'mm':
        mm = []
        for idx, val in enumerate(localkey_bigrams_list):
            preceding = val.split('_')[0]
            following = val.split('_')[1]
            if (preceding in MINOR_NUMERALS) and (following in MINOR_NUMERALS):
                mm.append(val)
        return mm


def compute_modulation_steps(partitioned_bigrams_list: List[str], partition_type: Literal['MM', 'Mm', 'mM', 'mm']):

    def get_key(my_dict, val):
        for key, value in my_dict.items():
            if val == value:
                return key

    modulation_steps_list = []
    # for idx, val in enumerate(partitioned_bigrams_list):
    #     preceding_RN = val.split('_')[0]
    #     following_RN = val.split('_')[1]

    # using the Enharmonic Pitch (12-TET semitones) in the pitchtype library
    if partition_type == 'MM':
        for idx, val in enumerate(partitioned_bigrams_list):
            preceding_RN = val.split('_')[0]
            following_RN = val.split('_')[1]

            preceding_SP = pitchtypes.EnharmonicPitchClass(get_key(my_dict=MAJOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS, val=preceding_RN))
            following_SP = pitchtypes.EnharmonicPitchClass(get_key(my_dict=MAJOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS, val=following_RN))

            interval = following_SP - preceding_SP
            modulation_steps_list.append(interval)
        return modulation_steps_list

    elif partition_type == 'Mm':
        for idx, val in enumerate(partitioned_bigrams_list):
            preceding_RN = val.split('_')[0]
            following_RN = val.split('_')[1]

            preceding_SP = pitchtypes.EnharmonicPitchClass(get_key(my_dict=MAJOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS, val=preceding_RN))
            following_SP = pitchtypes.EnharmonicPitchClass(get_key(my_dict=MINOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS, val=following_RN))

            interval = following_SP - preceding_SP
            modulation_steps_list.append(interval)
        return modulation_steps_list

    elif partition_type == 'mM':
        for idx, val in enumerate(partitioned_bigrams_list):
            preceding_RN = val.split('_')[0]
            following_RN = val.split('_')[1]

            preceding_SP = pitchtypes.EnharmonicPitchClass(get_key(my_dict=MINOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS, val=preceding_RN))
            following_SP = pitchtypes.EnharmonicPitchClass(get_key(my_dict=MAJOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS, val=following_RN))

            interval = following_SP - preceding_SP
            modulation_steps_list.append(interval)

        return modulation_steps_list

    elif partition_type == 'mm':
        for idx, val in enumerate(partitioned_bigrams_list):
            preceding_RN = val.split('_')[0]
            following_RN = val.split('_')[1]

            preceding_SP = pitchtypes.EnharmonicPitchClass(get_key(my_dict=MINOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS, val=preceding_RN))
            following_SP = pitchtypes.EnharmonicPitchClass(get_key(my_dict=MINOR_ROMAN_NUMERALS_TO_SPELLED_PITCHCLASS, val=following_RN))

            interval = following_SP - preceding_SP
            modulation_steps_list.append(interval)

        return modulation_steps_list


if __name__ == '__main__':
    localkey_bigrams = ['I_V', 'V_V', 'V_vi', 'vi_I', 'I_V', 'vi_i']

    MM = ['I_I', 'I_V', 'I_IV', 'IV_V', 'IV_I', 'V_I', 'V_V', 'bIII_I', 'I_bIII', 'I_III']
    mM = ['vi_I', 'iv_I', 'vi_V', 'iv_V', 'iii_I']

    result = compute_modulation_steps(partitioned_bigrams_list=mM, partition_type='mM')

    print(result)

