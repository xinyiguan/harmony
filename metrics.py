from typing import List

import pandas as pd
from pitchtypes import SpelledPitchClass, SpelledPitchClassArray, asic, aspc, SpelledIntervalClass

from harmonytypes.degree import Degree
from harmonytypes.key import Key
from harmonytypes.numeral import Numeral
from harmonytypes.quality import TertianHarmonyQuality

#   1. pitch class content index (out-of-context index, root as tonic)
##  1.1 sum (on line of fifths values) of all non-diatonic notes.
### m: PC->Z
"""
Example:
    chord = V7(+2) in F major, (root=C, mode=major)
    pcs ={'C', 'E', 'G', 'Bb', 'D'}
    m({Bb}) = -2
"""


def pc_content_index_m1(numeral: Numeral, reference_key: Key) -> float:
    # line of fifth value center on C
    non_diatonic_spcs = numeral.non_diatonic_spcs(reference_key=reference_key)
    index = sum([x.fifths() for x in non_diatonic_spcs])
    return index


def pc_content_index_m2(numeral: Numeral, reference_key: Key) -> float:
    # line of fifth value center on the reference tonic
    non_diatonic_spcs = numeral.non_diatonic_spcs(reference_key=reference_key)
    non_diatonic_scale_degrees = [reference_key.find_degree(x) for x in non_diatonic_spcs]
    index = sum([x.fifth(mode=reference_key.mode) for x in non_diatonic_scale_degrees])
    return index


def pc_content_index(numeral: Numeral, nd_ref_key: Key, d5th_ref_tone: SpelledPitchClass) -> float:
    non_diatonic_spcs = numeral.non_diatonic_spcs(reference_key=nd_ref_key)
    index = sum([d5th(reference_tone=d5th_ref_tone, other=x) for x in non_diatonic_spcs])
    return index


def d5th(reference_tone: SpelledPitchClass, other: SpelledPitchClass) -> int:
    interval: SpelledIntervalClass = reference_tone.interval_to(other=other)
    result = interval.fifths()
    return result


def within_chord_ci(numeral: Numeral) -> int:
    non_diatonic_spcs = numeral.non_diatonic_spcs(reference_key=numeral.key_if_tonicized())
    result = sum([d5th(reference_tone=numeral.key_if_tonicized().tonic, other=x) for x in non_diatonic_spcs])
    return result


def within_key_ci(reference_key: Key, root: SpelledPitchClass) -> int:
    result = d5th(reference_tone=reference_key.tonic, other=root)
    return result


def between_keys_ci(source_key: Key, target_key: Key) -> int:
    result = d5th(reference_tone=source_key.tonic, other=target_key.tonic)
    return result


def test():
    df: pd.DataFrame = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')

    df_row = df.iloc[1]
    numeral = Numeral.from_df(df_row)
    print(f'{numeral=}')
    result = pc_content_index_m2(numeral=numeral, reference_key=numeral.key_if_tonicized())
    print(f'non-diatonic-notes: {numeral.non_diatonic_spcs(reference_key=numeral.key_if_tonicized())}')
    print(f'{result=}')


def test1():
    numeral = Numeral(numeral_string="bII", global_key=Key.from_string("C"), local_key=Key.from_string("G"),
                      harmony_quality=TertianHarmonyQuality(asic(["M3", "m3"])), bass=SpelledPitchClass("Ab"),
                      root=SpelledPitchClass("Ab"), degree=Degree.from_string("b2"), spcs=aspc(["Ab", "C", "Eb"]))
    print(f'{numeral=}')

    ref_key = numeral.key_if_tonicized()
    print(f'non-diatonic-notes: {numeral.non_diatonic_spcs(reference_key=ref_key)}')
    index_m2 = pc_content_index_m2(numeral=numeral, reference_key=ref_key)
    print(f'{index_m2=}')

    new_pc = pc_content_index(numeral=numeral, nd_ref_key=ref_key, d5th_ref_tone=ref_key.tonic)
    print(f'{new_pc=}')


if __name__ == '__main__':
    test1()
