from typing import List

import numpy as np
import pandas as pd
from pitchtypes import SpelledPitchClass, SpelledPitchClassArray, asic, aspc, SpelledIntervalClass, EnharmonicPitchClass

from harmonytypes.degree import Degree
from harmonytypes.key import Key
from harmonytypes.numeral import Numeral
from harmonytypes.quality import TertianHarmonyQuality
from harmonytypes.scale import DiatonicScale, OctatonicScale, HexatonicScale

#   1. pitch class content index (out-of-context index, root as tonic)
##  1.1 sum (on line of fifths values) of all non-diatonic notes.
### m: PC->Z
"""
Example:
    chord = V7(+2) in F major, (root=C, mode=major)
    pcs ={'C', 'E', 'G', 'Bb', 'D'}
    m({Bb}) = -2
"""


def pc_content_index(numeral: Numeral, nd_ref_key: Key, d5th_ref_tone: SpelledPitchClass) -> float:
    non_diatonic_spcs = numeral.non_diatonic_spcs(reference_key=nd_ref_key)
    index = sum([d5th(reference_tone=d5th_ref_tone, other=x) for x in non_diatonic_spcs])
    return index


def d5th(reference_tone: SpelledPitchClass, other: SpelledPitchClass) -> int:
    interval: SpelledIntervalClass = reference_tone.interval_to(other=other)
    result = interval.fifths()
    return result


class MultilevelChordChromaticity:

    @staticmethod
    def within_chord_ci(numeral: Numeral) -> int:
        non_diatonic_spcs = numeral.non_diatonic_spcs(reference_key=numeral.key_if_tonicized())
        result = sum([d5th(reference_tone=numeral.key_if_tonicized().tonic, other=x) for x in non_diatonic_spcs])
        return result

    @staticmethod
    def within_key_ci(reference_key: Key, root: SpelledPitchClass) -> int:
        result = d5th(reference_tone=reference_key.tonic, other=root)
        return result

    @staticmethod
    def between_keys_ci(source_key: Key, target_key: Key) -> int:
        result = d5th(reference_tone=source_key.tonic, other=target_key.tonic)
        return result

    @staticmethod
    def chord_chromaticity_vector() -> np.ndarray:
        c_wc = MultilevelChordChromaticity.within_chord_ci()
        c_wk = MultilevelChordChromaticity.within_key_ci()
        c_bk = MultilevelChordChromaticity.between_keys_ci()
        v = np.array([c_wc, c_wk, c_bk])
        return v

    @staticmethod
    def multilevel_chord_chromaticity():
        raise NotImplementedError


class ChromaticIndex_Def2_Yannis:
    @staticmethod
    def within_chord_ci(numeral: Numeral) -> int:  # excluding the root
        raise NotImplementedError

    @staticmethod
    def within_key_ci(reference_key: Key, root: SpelledPitchClass) -> int:  # only the root
        raise NotImplementedError

    @staticmethod
    def between_keys_ci(source_key: Key, target_key: Key) -> int:
        raise NotImplementedError


# =====================================================================================================================

def out_of_scale_index(numeral: Numeral, scale: DiatonicScale | OctatonicScale | HexatonicScale) -> int:
    # diatonic does not assume enharmonic equivalence
    if isinstance(scale, DiatonicScale):
        out_of_scale_pitches = [x for x in numeral.spcs if not scale.check_membership(x)]
        penalty = sum([abs(d5th(reference_tone=scale.tonic, other=x)) for x in out_of_scale_pitches])

    # octatonic and hexatonic assume enharmonic equivalence
    elif isinstance(scale, OctatonicScale):
        numeral_epcs = [EnharmonicPitchClass(str(x)) for x in numeral.spcs]
        out_of_scale_pitches = [x for x in numeral_epcs if not scale.check_membership(x)]

        penalty = ...

    elif isinstance(scale, HexatonicScale):
        numeral_epcs = [EnharmonicPitchClass(str(x)) for x in numeral.spcs]
        out_of_scale_pitches = [x for x in numeral_epcs if not scale.check_membership(x)]

        penalty = ...


    else:
        raise NotImplementedError
    print(f'{scale.members=}')
    print(f'{penalty=}')
    return penalty


def test():
    df: pd.DataFrame = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')

    df_row = df.iloc[1]
    numeral = Numeral.from_df(df_row)
    print(f'{numeral=}')
    result = out_of_scale_index(numeral=numeral,
                                scale=DiatonicScale.from_tonic(tonic=SpelledPitchClass("D"), mode="major"))
    return result


def test1():
    numeral = Numeral(numeral_string="bII", global_key=Key.from_string("C"), local_key=Key.from_string("G"),
                      harmony_quality=TertianHarmonyQuality(asic(["M3", "m3"])), bass=SpelledPitchClass("Ab"),
                      root=SpelledPitchClass("Ab"), degree=Degree.from_string("b2"), spcs=aspc(["Ab", "C", "Eb"]))
    print(f'{numeral=}')

    ref_key = numeral.key_if_tonicized()
    print(f'non-diatonic-notes: {numeral.non_diatonic_spcs(reference_key=ref_key)}')

    new_pc = pc_content_index(numeral=numeral, nd_ref_key=ref_key, d5th_ref_tone=ref_key.tonic)
    print(f'{new_pc=}')


if __name__ == '__main__':
    test()
