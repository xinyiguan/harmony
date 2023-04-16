from typing import List

import pandas as pd
from pitchtypes import SpelledPitchClass, SpelledPitchClassArray, asic, aspc

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
    # non_diatonic_spcs = [x for x in numeral.spcs if x not in tonicized_key.get_scale()]
    non_diatonic_spcs = numeral.non_diatonic_spcs(reference_key=reference_key)
    index = sum([x.fifths() for x in non_diatonic_spcs])
    return index

def pc_content_index_m2(numeral: Numeral, reference_key:Key)->float:
    # line of fifth value center on the reference tonic
    non_diatonic_spcs = numeral.non_diatonic_spcs(reference_key=reference_key)
    non_diatonic_scale_degrees = [reference_key.find_degree(x) for x in non_diatonic_spcs]
    index = sum([x.fifth(mode=reference_key.mode) for x in non_diatonic_scale_degrees])
    return index


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
    numeral= Numeral(numeral_string="bII", global_key=Key.from_string("C"), local_key=Key.from_string("G"),
                     harmony_quality=TertianHarmonyQuality(asic(["M3", "m3"])), bass=SpelledPitchClass("Ab"),
                     root=SpelledPitchClass("Ab"), degree=Degree.from_string("b2"), spcs=aspc(["Ab", "C", "Eb"]))
    print(f'{numeral=}')

    ref_key = numeral.key_if_tonicized()
    print(f'non-diatonic-notes: {numeral.non_diatonic_spcs(reference_key=ref_key)}')
    result = pc_content_index_m2(numeral=numeral, reference_key=ref_key)

    print(f'{result=}')

if __name__ == '__main__':
    test1()
